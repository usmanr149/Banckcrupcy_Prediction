# IMPORTS
# ______________________________________________________________________________________________________________________
from DSI import *

# PARAMS
# ______________________________________________________________________________________________________________________
# LOCATIONS TO STORE THE DATA
LOCATION_OUTPUT = "DSI_output/"
# LOCATION_LOPUCKI = ""
# LOCATION_EDGAR = ""
LOCATION_LOPUCKI = "data/Florida-UCLA-LoPucki Bankruptcy Research Database 1-12-2023/Florida-UCLA-LoPucki Bankruptcy Research Database 1-12-2023.csv"
LOCATION_EDGAR = "location_edgar"
# GENERAL
begin_year = 2014 #in the paper data from 2000 untill 2020 was used
end_year = 2020
# deal with a subsample of the data (to speed things up and try out the code)
sample = True

# SCRIPT
# ______________________________________________________________________________________________________________________
print(100*'_')
print('INTEGRATION OF DATA SOURCES')
integrator = DSI(LOCATION_LOPUCKI=LOCATION_LOPUCKI, LOCATION_EDGAR=LOCATION_EDGAR)
failed, healthy = integrator.integrate(begin_year=begin_year, end_year=end_year)
print('STORING HEALTHY DF')
if sample:
   companies = random.sample(list(healthy['cik'].unique()), 5000)
   healthy = healthy[healthy['cik'].isin(companies)].reset_index(drop=True)
healthy.to_csv(LOCATION_OUTPUT + 'healthy_text_all.csv')
print('STORING FAILED DF')
failed.to_csv(LOCATION_OUTPUT + 'failed_text_all.csv')
# ______________________________________________________________________________________________________________________
print('STORING THE BANKRUPTCY YEARS')
BRD_years = integrator.store_BRD_years()
BRD_years.to_csv(LOCATION_OUTPUT + 'failure_years.csv')
print(100*'_')


