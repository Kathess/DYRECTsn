# DYnamic Reliable rEal-time Communication in Tsn (DYRECTsn)

Framework for TSN flow reservation, as described in the PhD Thesis "Reliable Real-Time Communication in Time-Sensitive 
Networking with Static and Dynamic Network Traffic" by Lisa Maile.

The framework implements two phase: 1) An optional 
[offline pre-configuration phase](https://link.springer.com/chapter/10.1007/978-3-031-48885-6_8) to determine 
the delay-budgets for each queue in the network. 2) All functions to dynamically add and remove flows to the network.

The focus of the framework is to ensure upper bounds for the flows delays at any time during the network 
operation using [Network Calculus](https://ieeexplore.ieee.org/abstract/document/9123308). Each new reservation only 
requires an update of the ports which the new flow traverses. Existing reservations are not violated. The concept can
be used in [centralized](https://ieeexplore.ieee.org/abstract/document/9913646) and
[decentralized](https://dl.acm.org/doi/abs/10.1145/3575757.3593644) network architecture and the implementation 
currently only supports Credit-Based Shaper networks.

### Input
The framework allows for a set of optional static flows as input, as well as estimations of the future
bandwidth which dynamic flows will use. Then the network is optimized on these inputs (this step
is optional if the user has own information about the desired per-hop delay). 

### Output
With these configuration the frame can add and remove flows during network operation. 
The framework determines path, the number of queues, flow priority, per-hop delay values 
and idleSlopes of the flows.
It considers worst-case delay bounds and buffer sizes and supports unicast, multicast and 
disjoint flows (= flows using FRER). For FRER flows, it also derives save configurations
for the sequence recovery function, [see](https://ieeexplore.ieee.org/document/9838905).

See especially 
[this publication](https://link.springer.com/chapter/10.1007/978-3-031-48885-6_8) for details, illustrations,
and an overview of the framework.

## Academic Attribution

If you use this library for research, please include the following reference in any resulting publication:

```plain
@inproceedings{DYRECTsn,
    author={Maile, Lisa and Hielscher, Kai-Steffen and German, Reinhard},
    editor={Kalyvianaki, Evangelia and Paolieri, Marco},
    title={{Combining Static and Dynamic Traffic with Delay Guarantees in Time-Sensitive Networking},
    booktitle={Performance Evaluation Methodologies and Tools},
    year={2024},
    publisher={Springer Nature Switzerland},
    address={Cham},
    pages={117--132},
    isbn={978-3-031-48885-6},
    doi={10.1007/978-3-031-48885-6\_8},
    url={https://link.springer.com/chapter/10.1007/978-3-031-48885-6_8}
}
```

If you use the configuration for the redundant data transmission implemented in this library, also include the reference:

```plain
@inproceedings{FRER_configuration_2022,
    author={Maile, Lisa and Voitlein, Dominik and Hielscher, Kai-Steffen and German, Reinhard},
    booktitle={ICC 2022 - IEEE International Conference on Communications}, 
    title={{Ensuring Reliable and Predictable Behavior of IEEE 802.1CB Frame Replication and Elimination}}, 
    year={2022},
    volume={},
    number={},
    pages={2706--2712},
    doi={10.1109/ICC45855.2022.9838905},
    url={https://ieeexplore.ieee.org/abstract/document/9838905}
}
```

## Usage

For convenience, a demo file was created which explains the main concepts of the library as comments in the code.

To run the demo file, follow the following steps:

1. To set the path, navigate to the uppermost folder (DYRECTsn) and type:
```
export PYTHONPATH=.
```
2. Then run the demo file and check the commentaries for more information:
```
python3 optimization/demo_call.py
```
Tested with Python 3.6 and PyCharm 2021.2.2.

## Report Bugs and other Issues

Reports on bugs and other issues are very welcome, especially due to many quite recent changes.
Please make sure to attach a minimal example demonstrating the bug/issue to your report.

## Contact

In case of questions, feel free to contact [Lisa Maile](mailto:lisa.maile@fau.de?subject=[DYRECTsn%20GitHub]%20).




