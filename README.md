## HAW Branch Notes
This branch contains scenarios and amendments for joint work of Lisa Maile and Timo Salomon (see Reference).
If you are interested in reproducing the results or checking out the scenarios please use this branch version (see Scenarios).
Otherwise, consider switching to the main branch for using the DYRECTsn framework itself. 

### Reference 
For a reference to the DYRECTsn framework check out the Academic Attribution.
If you use or refer to our scenarios, please include the following reference in any resulting publication:
> Timo Salomon, Lisa Maile, Philipp Meyer, Franz Korf, Thomas C. Schmidt, "Negotiating strict latency limits for dynamic real-time services in vehicular time-sensitive networks," Vehicular Communications, vol. 57, p. 100 985, Feb. 2026. DOI: 10.1016/j.vehcom.2025.100985
```bibtex
@Article{smmks-nslld-25,
  author    = {Timo Salomon and Lisa Maile and Philipp Meyer and Franz Korf and Thomas C. Schmidt},
  journal   = {Vehicular Communications},
  title     = {{Negotiating strict latency limits for dynamic real-time services in vehicular time-sensitive networks}},
  year      = {2026},
  month     = feb,
  pages     = {100985},
  volume    = {57},
  doi       = {10.1016/j.vehcom.2025.100985},
  publisher = {Elsevier},
}
```

### Scenarios
The paper has a CBS max latency study and a realistic in car network. See also additional readmes in scenario folders.
For our paper, we use a comprehensive evaluation environment that also relies on OMNeT++ simulation models.
Check the full [workspace repository](https://github.com/CoRE-RG/vehcom25-soa-strict-cbs-latency) for the VehCom 2025 paper evaluation setup.

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
    title={{Combining Static and Dynamic Traffic with Delay Guarantees in Time-Sensitive Networking}},
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

## Installation and Usage

For convenience, a demo file was created which explains the main concepts of the library as comments in the code.

To run the demo file, follow the following steps:

1. To install the required Python packages, please use the `requirements.txt` file. You can install these packages using pip:

```bash
pip install -r requirements.txt
```

2. To set the path, navigate to the uppermost folder (DYRECTsn) and type:
```
export PYTHONPATH=.
```
3. Then run the demo file and check the commentaries for more information:
```
python3 optimization/demo_call.py
```
Tested with Python 3.6, 3.9, 3.10, and 3.11.

This project includes a GitHub Action for simple testing. The action is configured to automatically run tests when changes are pushed to the repository or a pull request is opened.
You can find the configuration for this action in the `.github/workflows` directory.

## License
This project is licensed under the GNU Lesser General Public License - see the [LICENSE](LICENSE) file for details.

## Report Bugs and other Issues

Reports on bugs and other issues are very welcome, especially due to many quite recent changes.
Please make sure to attach a minimal example demonstrating the bug/issue to your report.

## Contact

In case of questions, feel free to contact [Lisa Maile](mailto:lisa.maile@fau.de?subject=[DYRECTsn%20GitHub]%20).



