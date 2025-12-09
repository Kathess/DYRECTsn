# Car Network

## References
This study is from the paper:
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

The original car network has been published in:
> Philipp Meyer, Timo Häckel, Teresa Lübeck, Franz Korf, Thomas C. Schmidt, A Framework for the Systematic Assessment of Anomaly Detectors in Time-Sensitive Automotive Networks, In: Proc. of the 15th IEEE Vehicular Networking Conference (VNC), pp. 57-64, IEEE, May 2024.
```Bibtex
@inproceedings{mhlks-fsaad-24,
  author = {Philipp Meyer and Timo H{\"a}ckel and Teresa L{\"u}beck and Franz Korf and Thomas C. Schmidt},
  title = {{A Framework for the Systematic Assessment of Anomaly Detectors in Time-Sensitive Automotive Networks}},
  booktitle = {Proc. of the 15th IEEE Vehicular Networking Conference (VNC)},
  pages = {57-64},
  location = {Kobe, Japan},
  month = {May},
  year = {2024},
  publisher = {IEEE},
  doi = {10.1109/VNC61989.2024.10576017},
}
```

## Scenario
This scenario relies on the OMNeT++ simulation setup from our [workspace repository](https://github.com/CoRE-RG/vehcom25-soa-strict-cbs-latency) for the VehCom 2025 paper.
The network represents a realistic in-car network transformed into a future zonal topology with realistic traffic patterns. 
We compare different reservation schemes for PCP 4 and 5 CBS shaping, one of them implemented in DYRECTsn -- the delay budget approach. 

### Traffic

| Traffic                            | Source                        | Destination                   | VLAN  | Priority | Shaping           | Redundancy          | Layer 3+ |
|------------------------------------|-------------------------------|-------------------------------|:-----:|:--------:|-------------------|---------------------|----------|
| gPTP Sync                          | (masterClock)                 | -                             | 0     | 7        | -                 | 2 gPTP Domains      | -        |
| Manual Throttle / Brake / Steer    | zonalControllerFrontLeft      | zonalController*              | 0/1/2 | 6        | 802.1Q GCL Window | 802.1CB in Backbone | IP/UDP   |
| Automatic Throttle / Brake / Steer | adas                          | zonalController*              | 0/1/2 | 6        | 802.1Q GCL Window | 802.1CB in Backbone | IP/UDP   |
| Video (2 Source Apps)              | camera*                       | adas                          | 0/1/2 | 5        | 802.1Q CBS        | 802.1CB in Backbone | IP/UDP   |
| LIDAR (4 Source Apps)              | lidar*                        | adas                          | 0/1/2 | 5        | 802.1Q CBS        | 802.1CB in Backbone | IP/UDP   |
| Control (201 Source Apps)          | zonalController*/infotainment | zonalController*/infotainment | 0     | 4        | 802.1Q CBS        | -                   | IP/UDP   |
| V2X                                | connectivityGateway/adas      | connectivityGateway/adas      | 0     | 2        | -                 | -                   | IP/TCP   |
| Background                         | *                             | *                             | 0     | 0        | -                 | -                   | *        |

## Configurations
This DYRECTsn scenario produces the Network Calculus (NC) idle slope configurations and worst case latency analysis for the in-car network.
When running the OMNeT simulation, a configuration json-file will be created such as "flowUpdatesStreamCMI.json" which contains all flow installations during this run. 
This can be fed into DYRECTsn to run a worst-case analysis and to identify Idle Slopes that meet the deadlines. 
```parse_omnet_output.ipynb``` can be used to convert this json output int car_flows.py and the car_topology.py which will then be used for the scenario. 

## Study Execution and Evaluation
The analysis was done in Ubuntu 22.04 (WSL).

### Run Studies

```find_best_solution.sh``` will execute the ```car_run_scenario.py``` for 16*5 times to find a good configuration for delay budgets using the meta-heuristic in DYRECTsn. A working configuration for the delay budgets has been pre-configured in the scenario.

```run_study.sh``` will run the scenarios. Execute from this directory, creates a results folder. You can also just call ```car_run_scenario.py``` with python directly.
