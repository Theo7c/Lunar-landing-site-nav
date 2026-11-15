from google import genai


def GeneratePrefix():
    prefix = """You are a NASA supercomputer who excells at generating
    mission parameters for setting up a moon base. The four
    metrics to be optimised are: Terrain Safety Score, Rover Traverse Science Score,
    Solar Illumination Visibility Score, Earth Line-of-Sight Communication
    Score. Refer to the weights of these scores as w_safety, w_science,
    w_sun, and w_earth respectively.
    w_safety: The safety score evaluates whether a landing
    site satisfies spacecraft touchdown stability constraints based on
    local slope and surrounding surface roughness.
    w_science: The science score quantifies the cumulative scientific
    value a surface rover can collect from nearby Sites of Scientific
    Interest (SSIs) within a maximum distance.
    w_sun and w_earth: Both astronomical visibility scores measure the
    fraction of time over a mission window that the target body
    (Sun for solar power and thermal stability; Earth for direct-to-Earth
    line-of-sight communications) clears the surrounding local lunar topography.
    I need you to compute a set of four weights for these scores to prioritise
    different aspects of the lunar mission based on a prompt which will
    be provided, in order to determine which site would be the most appropriate
    for each specific mission. These weights must be in the range [0,1].
    You must not calculate the scores themselves from any data; these will
    be determined separately.
    To give an example, if the prompt states that nuclear reactors are
    going to be used in the mission then solar power should be weighted
    lower than other metrics.
    The response you give must consist of the four values of weights first
    followed by a short justification for each (no more than two sentences).
    The weights must be returned in a tuple, with brackets surrounding the
    values and commas between them, in the order
    (w_safety, w_science, w_sun, and w_earth). An example response here is
    (0.0,0.5,0.13,0.932).
    Justifications should be formatted in the style Science: justification
    of science weight.
    Finally you should also provide a recommended landing site for this
    mission by specifying the lattitude and longitude of the site in the form
    (lat,long) in between the weights and the justifications sections in
    your response. Do not include cardinal directions, only numerical values.
    The prompt from the NASA engineer is:"""
    prefix = prefix.replace("\n", " ").replace("    ", "")
    return prefix

def GenerateDefaultQuery():
    safetyPrompt = "We have created a new propulsion sytem to aid in touchdown on the lunar surface so the safety aspect is not as important here. "
    #safety weight is low
    sciencePrompt = " "
    #science weight is not stated
    sunPrompt = "The new systems we are testing require a large amount of energy to power, which we are planning on harvesting from the Sun. "
    #sun weight is high
    earthPrompt = "It is essential we transmit our results back to Earth, however this can be done at any time during the large mission window. "
    #earth weight must be interpreted - high or low
    query = safetyPrompt + sciencePrompt + sunPrompt + earthPrompt
    return query

def GetAIResponse(APIKey:str, query:str, prefix=""):
    client = genai.Client(api_key=APIKey)
    response = client.interactions.create(
        model="gemini-3.5-flash-lite",
        input=prefix+"\n"+query)
    return response.output_text

def GetTupleOfFloats(text: str):
    lbracket = 0
    rbracket = 0
    for num, char in enumerate(text):
        if char == "(":
            lbracket = num
        elif char == ")":
            rbracket = num
            break
    valuesStr = text[lbracket+1:rbracket].split(",")
    values = []
    for v in valuesStr:
        values.append(float(v))
    return values, rbracket

def FormatResponse(response:str):
    weights, weightEndIndex = GetTupleOfFloats(response.strip("\n"))
    coords, coordEndIndex = GetTupleOfFloats(response[weightEndIndex+1:].strip("\n"))
    justifications = response[weightEndIndex+coordEndIndex+4:].strip("\n")
    return weights, coords, justifications


if __name__ == "__main__":
    prefix = GeneratePrefix()
    query = GenerateDefaultQuery()
    response = GetAIResponse(APIKey, query, prefix)
#     response = """(0.1, 0.4, 0.9, 0.5)

# (-89.9, 0.0)

# Safety: Advanced propulsion significantly mitigates touchdown risks, allowing us to minimize the safety weight despite local terrain challenges.
# Science: A robust scientific return remains a primary mission objective, balancing the exploration of nearby sites with power and communication constraints.
# Sun: High solar illumination is absolutely critical to harvest the massive amounts of energy required to power the new propulsion and operational systems.
# Earth: Direct-to-Earth communication is important for transmitting results, but a moderate weight suffices since transmissions can occur flexibly throughout the large mission window."""
    weights, coords, justifications = FormatResponse(response)