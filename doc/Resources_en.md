# Notes on System Resource Estimation for Plato

The following settings should fit most algorithms supported in Plato. For exceptions, please refer to the specific algorithm documentation.

### Number of Instances

To simply put: The smaller, the better.

Use as smaller number of instances as possible given enough number of threads and memory for one instance. The reason is that distributed graph algorithm will partition the whole graph into different instances, a smaller number of instance leads to smaller communication cost.

### Number of Threads within One Instance

To simply put: The more, the better.

Users of exclusive clusters can consider using this setting: One instance per machine, N threads per instance (N is the number of cores of the CPU). Ninja users can consider using this setting: N instance per machine (N being the number of NUMA-node), evenly distribute the number of threads (can even bind specific TIDs to specific processors for efficient execution).

### Memory per Instance

To simply put: 2 * Original Graph Size in Bytes / Number of Instances
The reason we double the original graph size is to guarantee algorithm run normally with peak memory use.

### An Example

For example, to run a graph algorithm on a weightless graph with 2000 billion edges, then the total memory should be: 200G * 8 (8 bytes per edge) * 2 = 3.2TBytes. If every instance acquires 100GBytes memory, there at least need to be 32 instances.
