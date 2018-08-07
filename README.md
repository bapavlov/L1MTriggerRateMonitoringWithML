# L1MTriggerRateMonitoringWithML
Monitoring the L1Muon trigger rates using ML/DL techniques

In order to extract the instantaneous luminosity, please, use the brilcalc tool described here:

https://cms-service-lumi.web.cern.ch/cms-service-lumi/brilwsdoc.html

and the following command to have instantaneous luminosity vs. lumisections:

```
brilcalc lumi --byls -u hz/ub -r 306777 -o lumi.csv --output-style csv
```

To collect the informations about the trigger rates, please, use the python script in the "rates" folder:

```
python tm_mon.py 306777
```

it will generate a csv file.

For training and test the NNs, please, use the jupyter notebook. It takes the two csv files as input.
