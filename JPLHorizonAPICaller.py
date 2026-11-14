from pandas import read_csv
from requests import get
from io import StringIO

def GetHorizonsEphemeris(target:str, coords:tuple, timeRange:tuple, timeResolution:tuple):
    """
    Summary
    -------
    Returns DataFrame containing preset ephemeris data using parameters.

    Parameters
    ----------
    target : str
        Earth or Sun.
    coords : tuple of floats, ints, or strings
        (longitude,latitude,elevation). deg,deg,km relative to avg radius.
    timeRange : tuple of strings
        (startTime,stopTime) in YYYY-MM-DD {HH:MM}. {} indicates optional parameters.
    timeResolution : tuple of (string/float/int, string)
        (quantity,unit). eg (1, day)

    Returns
    -------
    DataFrame of (dateTime,SunVisible,EarthVisible,azimuth,altitude)
    """
    try:
        if target == "Earth":
            command = 399
        elif target == "Sun":
            command = 10
    except:
        raise NotImplementedError("Only Earth and Sun are valid targets")
    try:
        longitude, latitude, altitude = coords
        startTime, stopTime = timeRange
        timeQuantity, timeUnits = timeResolution
    except:
        raise TypeError("Check parameters are in the correct form")
    parameters = """MAKE_EPHEM=YES
EPHEM_TYPE=OBSERVER
CENTER='coord@301'
COORD_TYPE=GEODETIC
QUANTITIES='4'
REF_SYSTEM='ICRF'
CAL_FORMAT='CAL'
CAL_TYPE='M'
TIME_DIGITS='MINUTES'
ANG_FORMAT='HMS'
APPARENT='AIRLESS'
RANGE_UNITS='AU'
SUPPRESS_RANGE_RATE='NO'
SKIP_DAYLT='NO'
SOLAR_ELONG='0,180'
EXTRA_PREC='NO'
R_T_S_ONLY='NO'
CSV_FORMAT='YES'
OBJ_DATA='YES'"""
    apiParametersList = parameters.replace("\n","&")
    apiParametersList += "&COMMAND='"+str(command)+"'"
    apiParametersList += "&SITE_COORD='"+str(longitude)+","+str(latitude)+","+str(altitude)+"'"
    apiParametersList += "&START_TIME='"+startTime+"'"
    apiParametersList += "&STOP_TIME='"+stopTime+"'"
    apiParametersList += "&STEP_SIZE='"+str(timeQuantity)+" "+timeUnits.upper()+"S"+"'"
    apiParametersList += "&format=text"
    apiBaseURL = "https://ssd.jpl.nasa.gov/api/horizons.api?"
    ephemerisRequest = get(apiBaseURL + apiParametersList)
    ephemerisLines = ephemerisRequest.text.split("\n")
    SOE = 0
    EOE = -1
    for num, line in enumerate(ephemerisLines):
        if "$$SOE" in line:
            SOE = num
        elif "$$EOE" in line:
            EOE = num
    ephemerisData = ephemerisLines[SOE+1:EOE]
    ephemerisClean = []
    for line in ephemerisData:
        line = line[:len(line)-1]
        ephemerisClean.append(line)
    ephemerisFile = StringIO("\n".join(ephemerisClean))
    headers = ["dateTime","SunVisible","EarthVisible","azimuth","altitude"]
    data = read_csv(ephemerisFile, sep=",", names=headers)
    return data
    
if __name__ == "__main__":
    target = "Earth"
    coords = (26,3.6,-1.9)
    timeRange = ("2006-01-01 12:45","2006-02-01 04:53")
    timeResolution = 1,"day"
    data = GetHorizonsEphemeris(target,coords,timeRange,timeResolution)
    
    
    
    
    
    
    