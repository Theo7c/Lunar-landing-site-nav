from JPLHorizonAPICaller import GetHorizonsEphemeris
import numpy as np

def FindBodyHorizonAngles(horizonAlt_Az:np.array, bodyAzAlt_time: np.array):
    #returns angle between astronomical body and local horizon over time
    #horizonAlt_Az: array of 360 values of local horizon in each direction
    #SunAzAlt_time: array of tf-ti values of body alt in direction az
    #make sure az goes 0->360 not -180->180
    bodyAz_time, bodyAlt_time = bodyAzAlt_time
    times = len(bodyAz_time) #tf-ti
    azimuths = len(horizonAlt_Az) #360
    bodyHorizonAngles = np.zeros(times)
    for time in range(times): #tf-ti iterations
        bodyAz = bodyAz_time[time]
        bodyAlt = bodyAlt_time[time]
        bodyAzInt = np.round(bodyAz)
        for azimuth in range(azimuths): #360 iterations
            if azimuth == bodyAzInt:
                horizonAlt = horizonAlt_Az[azimuth]
                angleDifference = bodyAlt - horizonAlt
                bodyHorizonAngles[time] = angleDifference
    return bodyHorizonAngles

def CalculateScore(bodyHorizonAngles:np.array):
    timeAboveHorizon = 0
    for angle in bodyHorizonAngles:
        if angle > 0:
            timeAboveHorizon += 1
    score = timeAboveHorizon/bodyHorizonAngles.size
    return score


if __name__ == "__main__":
    target = "Sun"
    coords = (26,3.6,-1.9)
    timeRange = ("2006-03-01","2006-04-01")
    timeResolution = 1,"day"
    data = GetHorizonsEphemeris(target, coords, timeRange, timeResolution)
    
    bodyAz = data["azimuth"]
    bodyAlt = data["altitude"]
    bodyAzAlt_time = (bodyAz,bodyAlt)
    
    horizonAlt_Az = np.zeros(360)
    for i in range(360):
        horizonAlt_Az[i] = 50*np.random.uniform()
    
    bodyHorizonAngles = FindBodyHorizonAngles(horizonAlt_Az, bodyAzAlt_time)
    score = CalculateScore(bodyHorizonAngles)
    print(score)






