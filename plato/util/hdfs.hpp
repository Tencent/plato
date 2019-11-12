/*
  Tencent is pleased to support the open source community by making
  Plato available.
  Copyright (C) 2019 THL A29 Limited, a Tencent company.
  All rights reserved.

  Licensed under the BSD 3-Clause License (the "License"); you may
  not use this file except in compliance with the License. You may
  obtain a copy of the License at

  https://opensource.org/licenses/BSD-3-Clause

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" basis,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
  implied. See the License for the specific language governing
  permissions and limitations under the License.

  See the AUTHORS file for names of contributors.
*/

#ifndef __PLATO_HDFS_HPP__
#define __PLATO_HDFS_HPP__

#include <cstdint>
#include <cstdlib>

#include <map>
#include <mutex>
#include <vector>
#include <memory>

#include "omp.h"
#include "hdfs.h"
#include "glog/logging.h"

#include "boost/format.hpp"
#include "boost/iostreams/stream.hpp"
#include "boost/iostreams/filter/gzip.hpp"
#include "boost/iostreams/filtering_stream.hpp"

namespace plato {

#ifndef UNUSED
#define UNUSED(x) (void)(x)
#endif

class hdfs_t {
public:
    /** the primary filesystem object */
    hdfsFS filesystem_;

public:
    /** hdfs file source is used to construct boost iostreams */
    class hdfs_device_t {
    public: // boost iostream concepts
        typedef char char_type;
        struct category:
            public boost::iostreams::closable_tag,
            public boost::iostreams::multichar_tag,
            public boost::iostreams::bidirectional_device_tag { };
            // public boost::iostreams::seekable_device_tag { };

    private:
        bool writable_;

        hdfsFS   filesystem_;
        hdfsFile file_;
        tOffset  fsize_;

    public:
        hdfs_device_t(): writable_(false), filesystem_(nullptr), file_(nullptr), fsize_(0) { }
        hdfs_device_t(const hdfs_t& hdfs_fs, const std::string& filename, const bool write = false):
            writable_(write), filesystem_(hdfs_fs.filesystem_), fsize_(0) {
            if (nullptr == filesystem_) { hdfs_failure("filesystem is nullptr"); }

            // open the file
            const int   flags = write? O_WRONLY : O_RDONLY;
            const int   buffer_size = 0; // use default
            const short replication = 0; // use default
            const tSize block_size = 0;  // use default;
            file_ = hdfsOpenFile(filesystem_, filename.c_str(), flags, buffer_size,
                                replication, block_size);
            if (nullptr == file_) {
                hdfs_failure((boost::format("can not open %s") % filename.c_str()).str());
            }

            hdfsFileInfo* pInfo = hdfsGetPathInfo(filesystem_, filename.c_str());
            if (nullptr == pInfo) {
                PLOG(WARNING) << boost::format("hdfsGetPathInfo failed(%s)") % filename.c_str();
            } else {
                fsize_ = pInfo->mSize;
                hdfsFreeFileInfo(pInfo, 1);
            }
        }

        void close(std::ios_base::openmode mode = std::ios_base::openmode()) {
            UNUSED(mode);
            if(nullptr == file_) return;
            if(hdfsFileIsOpenForWrite(file_)) {
                if (0 != hdfsFlush(filesystem_, file_)) {
                    hdfs_failure("flush hdfs-file failed");
                }
            }
            if (0 != hdfsCloseFile(filesystem_, file_)) {
                hdfs_failure("close hdfs-file failed");
            }

            file_ = nullptr;
        }

        /** the optimal buffer size is 0. */
        inline std::streamsize optimal_buffer_size() const {
            return 0;
        }

        std::streamsize read(char* strm_ptr, std::streamsize n) {
            return hdfsRead(filesystem_, file_, strm_ptr, n);
        } // end of read

        std::streamsize write(const char* strm_ptr, std::streamsize n) {
            return hdfsWrite(filesystem_, file_, strm_ptr, n);
        }

        // XXX seek is buggy, you should avoid to use it
        boost::iostreams::stream_offset seek(boost::iostreams::stream_offset off, std::ios_base::seekdir way) {
            if (hdfsFileIsOpenForWrite(file_)) {
                throw std::invalid_argument("hdfs file open for write can not seek location");
            }

            boost::iostreams::stream_offset next;
            switch (way) {
            case std::ios_base::beg:
                next = off;

                break;
            case std::ios_base::cur:
            {
                tOffset currentOff = hdfsTell(filesystem_, file_);
                if (-1 == currentOff) {
                    hdfs_failure("hdfs tell failed");
                }
                next = currentOff + off;

                break;
            }
            case std::ios_base::end:
                next = fsize_ + off - 1;

                break;
            default:
                throw std::ios_base::failure("bad seek direction");
            }

            if ((next < 0) || (next >= fsize_)) {
                throw std::invalid_argument("bad seek offset");
            }

            if (0 != hdfsSeek(filesystem_, file_, next)) {
                hdfs_failure("hdfs seek failed!");
            }

            return next;
        }

        bool good(void) const {
            return nullptr != file_;
        }
    }; // end of hdfs device

    /** hdfs dir source is used to construct boost iostreams */
    class hdfs_bunch_device_t {
    public: // boost iostream concepts
        typedef char char_type;
        struct category:
            public boost::iostreams::source_tag,
            public boost::iostreams::closable_tag,
            public boost::iostreams::multichar_tag { };

    private:
        hdfsFS                   filesystem_;
        hdfsFile                 file_;
        std::vector<std::string> filenames_;

    public:
        hdfs_bunch_device_t(): filesystem_(nullptr), file_(nullptr) { }
        hdfs_bunch_device_t(const hdfs_t& hdfs_fs, const std::vector<std::string>& fnames):
            filesystem_(hdfs_fs.filesystem_), file_(nullptr), filenames_(fnames) {
            if (nullptr == filesystem_) { hdfs_failure("filesystem is nullptr"); }
            if (filenames_.empty()) { LOG(WARNING) << "empty fnames"; }
        }

        void close(std::ios_base::openmode mode = std::ios_base::openmode()) {
            UNUSED(mode);
            if ((nullptr != file_) && (0 != hdfsCloseFile(filesystem_, file_))) {
                hdfs_failure("close hdfs-file failed");
            }
            file_ = nullptr;
            filenames_.clear();
        }

        /** the optimal buffer size is 0. */
        inline std::streamsize optimal_buffer_size() const {
            return 0;
        }

        std::streamsize read(char* strm_ptr, std::streamsize n) {
            auto move_next = [this](void) -> bool {
                if (filenames_.empty()) { return false; }

                std::string name = filenames_.back();
                filenames_.pop_back();

                file_ = hdfsOpenFile(filesystem_, name.c_str(), O_RDONLY, 0, 0, 0);
                if (nullptr == file_) {
                    hdfs_failure((boost::format("can not open %s") % name.c_str()).str());
                }
                return true;
            };

            if (nullptr == file_) {
                if (false == move_next()) { return 0; }
            }

            while (true) {
                tSize bytes = hdfsRead(filesystem_, file_, strm_ptr, n);

                switch (bytes) {
                case 0:
                    if (false == move_next()) { return 0; }
                    break;
                default:
                    return bytes;
                }
            }

            LOG(FATAL) << "forbidden area, something wrong happened!";
        } // end of read

        bool good(void) const {
            return nullptr != file_;
        }
    }; // end of hdfs device

    /**
     * The basic file type has constructor matching the hdfs device.
     */
    typedef boost::iostreams::stream<hdfs_device_t> fstream;
    typedef boost::iostreams::stream<hdfs_bunch_device_t> fbstream;

    /**
     * Open a connection to the filesystem. The default arguments
     * should be sufficient for most uses
     */
    hdfs_t(const std::string& host = "default", tPort port = 0 /* deprecated */) {
        auto builder = hdfsNewBuilder();
        hdfsBuilderSetNameNode(builder, host.c_str());

        if (std::getenv("HDFS_USER_NAME")) {
          hdfsBuilderSetUserName(builder, std::getenv("HDFS_USER_NAME"));
        }

        if (std::getenv("HADOOP_JOB_UGI")) {
          hdfsBuilderConfSetStr(builder, "hadoop.job.ugi", std::getenv("HADOOP_JOB_UGI"));
        }

        filesystem_ = hdfsBuilderConnect(builder);
        if (nullptr == filesystem_) { hdfs_failure("filesystem_ is nullptr"); }
    } // end of constructor

    ~hdfs_t() {
        if (0 != hdfsDisconnect(filesystem_)) { hdfs_failure("hdfsDisconnect failed"); }
    } // end of destructor

    /*
     * list files in 'path'
     *
     * \param path        hdfs path
     * \param not_empty   filter out empty file or not
     *
     * \return  files' name in 'path'
     **/
    inline std::vector<std::string> list_files(const std::string& path, bool not_empty = false) {
      int num_files = 0;
      hdfsFileInfo* hdfs_file_list_ptr = hdfsListDirectory(filesystem_, path.c_str(), &num_files);

      // copy the file list to the string array
      std::vector<std::string> files;
      for(int i = 0; i < num_files; ++i) {
        if (false == not_empty || hdfs_file_list_ptr[i].mSize > 0) {
          files.emplace_back(std::string(hdfs_file_list_ptr[i].mName));
        }
      }

      // free the file list pointer
      hdfsFreeFileInfo(hdfs_file_list_ptr, num_files);
      return files;
    } // end of list_files

    tOffset file_size(const std::string& path) {
        hdfsFileInfo* pInfo = hdfsGetPathInfo(filesystem_, path.c_str());

        if (nullptr == pInfo) {
            return -1;
        } else {
            return pInfo->mSize;
        }
    }

    int createDirectory(const std::string &path) {
      return hdfsCreateDirectory(filesystem_, path.c_str());
    }

    /**
     * Delete file.
     * @param path The path of the file.
     * @param recursive if path is a directory and set to
     * non-zero, the directory is deleted else throws an exception. In
     * case of a file the recursive argument is irrelevant.
     * @return Returns 0 on success, -1 on error.
     */
    int remove(const std::string &path, int recursive) {
      return hdfsDelete(filesystem_, path.c_str(), recursive);
    }

    /**
     * Move file from one filesystem to another.
     * @param srcFS The handle to source filesystem.
     * @param src The path of source file. 
     * @param dstFS The handle to destination filesystem.
     * @param dst The path of destination file. 
     * @return Returns 0 on success, -1 on error. 
     */
    int move(const char* src, hdfs_t& dstFS, const char* dst) {
      return hdfsMove(filesystem_, src, dstFS.filesystem_, dst);
    }

    /**
     * Checks if a given path exists on the filesystem
     * @param path The path to look for
     * @return Returns true on success, false on error.
     */
    bool exists(const std::string &path) {
      return hdfsExists(filesystem_, path.c_str()) == 0;
    }

    static std::string get_nm_from_path(const std::string& path);

    // get default hdfs instance
    static hdfs_t& get_hdfs(void);
    // get hdfs instance based on path
    static hdfs_t& get_hdfs(const std::string& path);

    /*
     * parse hdfs files as csv format
     *
     * \param fs        hdfs instance
     * \param chunks    hdfs file list
     * \param callback  user provide callback function that take csv lines as input
     *                  [[field_1, field_2, ...], ...]
     */
    static void parse_csv_files(const hdfs_t& fs, const std::vector<std::string>& chunks,
        std::function<void(const std::vector<std::vector<std::string>>&)> callback);

protected:
    static std::mutex mtx4inst_ ;
    static std::map<std::string, std::shared_ptr<hdfs_t>> instances_;

    static inline void hdfs_failure(const std::string& errmsg) {
        PLOG(ERROR) << errmsg;
        throw std::ios_base::failure(errmsg);
    }

    static inline void hdfs_failure(const char* sErrmsg) {
        PLOG(ERROR) << sErrmsg;
        throw std::ios_base::failure(sErrmsg);
    }

}; // end of class hdfs

}; // end of namespace plato

#endif

