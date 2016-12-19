# parallelizeddijkstra-
An attempt to parallelize Dijkstra's Algorithm on the GPU using CUDA C.

In this project instead of trying to find the shortest path between one source node to the rest of the graph,
we run dijkstra's algorithm on N cores and pass every node as the source node giving us the shortest path from ALL nodes
to the rest of the graph (For N being the number of nodes in the graph.)

We also compare the parallelized Dijkstra on the GPU to a GPU implementation of the Floyd-Warshall Algorithm which
tries to achieve the same goal of finding ALL shortest paths between every node. 

The Floyd-Warshall Algorithm was found at https://github.com/akintsakis/apspFloydWarshallCuda.


![Alt text] ()
![Alt text]()


## Getting Started

Clone the repo: git clone https://github.com/lramo062/parallelizeddijkstra-
cd into the directory and run make: $ make
to run dijkstra on both the cpu and gpu run: ./dijk
to run floy-warshal run: ./floyd

You can play around with the code to display the random generated graph as well as the result graph:




### Prerequisites

You will need a Nvidia Graphics card installed on your computer along with Cuda, make, & gcc.


### Installing

Clone https://github.com/lramo062/parallelizeddijkstra-


## Deployment

I plan to continue to work on this project and multithread dijkstra's algorithm.


## Contributing

Feel free to contact be about contributing to the project!

## Authors

* **Lester Ramos** - *Initial work* - [lramo062](https://github.com/lramo062)

See also the list of [contributors](https://github.com/lramo062/parallelizeddijkstra-/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Those who contribute to C CUDA documentation
