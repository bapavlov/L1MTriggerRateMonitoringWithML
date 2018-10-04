# L1MTriggerRateMonitoringWithML
Monitoring the L1Muon trigger rates using ML/DL techniques

In order to extract the instantaneous luminosity, please, use the brilcalc tool described here:

https://cms-service-lumi.web.cern.ch/cms-service-lumi/brilwsdoc.html

and the following command to have instantaneous luminosity vs. lumisections:

```
brilcalc lumi --byls -u hz/ub -r 306777 -o lumi.csv --output-style csv --type hfet
```

N.B.: Possible luminometers are hfoc, hfet, bcm1f, bcm1fsi, bcm1futca, pltzero, dt, pxl.

In order to check the 2017/2018 certified runs, please, use:

```
brilcalc lumi -i /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PromptReco/Cert_294927-306462_13TeV_PromptReco_Collisions17_JSON.txt -u /pb -o runs_2017.csv --output-style csv

brilcalc lumi -i /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/PromptReco/Cert_314472-321461_13TeV_PromptReco_Collisions18_JSON.txt -u /pb -o runs_2017.csv --output-style csv
```

To collect the informations about the trigger rates, please, use the python script in the "rates" folder:

```
python tm_mon.py 306777
```

it will generate a csv file.

For training and test the NNs, please, use the jupyter notebook. It takes the two csv files as input.
