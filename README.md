# Pyscript for Spot price prediction

> A multi-linear regression model to predict spot energy prices in the netherlands.
> Uses NED.nl data (wind, solar, consumption), time of day and historic prices to predict new prices. 

---

## Installation
It is assumed you have already Pyscript: https://github.com/custom-components/pyscript

### 1. Enable AppDaemon Apps in HACS

Because we are utilizing the AppDaemon category to cleanly route the folders, we need to ensure HACS is tracking that category.  
1. In Home Assistant, go to Settings -> Devices & Services.  
2. Find the HACS card and click Configure.
3. Check the box that says Enable AppDaemon apps discovery & tracking.
4. Click Submit. (HACS will quickly reload).Add the repository to Home Assistant

### 2. Link HACS to Your Repository

Now, let's link your private or public GitHub repository straight into HACS.
1. Go to the HACS panel from your Home Assistant sidebar.
2. Click the three dots (...) in the top right-hand corner and choose Custom repositories.
3. Fill out the fields:
	- Repository: Paste the full GitHub URL (e.g., [https://github.com/abrahamderonde/pyscript_price_forecast).
4. Category: Select AppDaemon.
5. Click Add.

### 3. The "Symlink Trick" (One-Time Setup)
By default, the AppDaemon category downloads your files into /config/appdaemon/apps/my_scripts/.

To make Home Assistant's PyScript integration read them seamlessly, we just need to create a symbolic link (a shortcut folder) using your Home Assistant terminal.

1. Open your Advanced SSH & Web Terminal (or use the Samba share to look at your folders).
2. Run this command to link the HACS download folder directly to your PyScript folder:
		Bash
		   ln -s /config/appdaemon/apps/my_scripts /config/pyscript
Now, whenever HACS downloads or updates your files into the AppDaemon folder, they instantly mirror right inside /config/pyscript/ where PyScript can read them!

---

## Available services

* ned_mlr_train_long
* ned_mlr_bootstrap_archive
* ned_mlr_backtest_forecasted
* ned_mlr_predict_7d
* ned_mlr_reset_forecast_archive

Pyscript services (functions) can be called in Home Assistant. 
example automation:
\`\`\`python
alias: NED MLR Predict
description: ""
triggers:
  - trigger: time
    at: "01:00:00"
conditions: []
actions:
  - action: pyscript.ned_mlr_predict_7d
    data: {}
mode: single
\`\`\`

## Workflow

* ned_mlr_train_long
At least run this before anything else. Recommended to run this on a regular basis
* ned_mlr_bootstrap_archive
optional. This will fill the forecast archive, when no data is generated yet
* ned_mlr_backtest_forecasted
Test the accuracy of your model. Predict function will use this to give the estimated accuracy band
* ned_mlr_predict_7d
This is the main function. I run this every day. See example above. 
* ned_mlr_reset_forecast_archive
If you want to you can clear the archive. 

---

*Built for personal use. Contributions welcome.*