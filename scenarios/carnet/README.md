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

## Dyrectsn scenarios
This scenario relies on the OMNeT++ simulation setup from our [workspace repository](https://github.com/CoRE-RG/vehcom25-soa-strict-cbs-latency) for the VehCom 2025 paper.
When running the OMNeT simulation, a configuration json-file will be created such as "flowUpdatesStreamCMI.json" which contains all flow installations during this run. 
This can be fed into DYRECTsn to run a worst-case analysis and to identify Idle Slopes that meet the deadlines. 

```parse_omnet_output.ipynb``` can be used to convert this json output int car_flows.py and the car_topology.py which will then be used for the scenario. 

The output in the results directory can then be used again in the omnetpp simulation.
