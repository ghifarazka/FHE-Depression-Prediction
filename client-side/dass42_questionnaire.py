"""
dass42_questionnaire.py

This file defines the questions and options for the DASS-42 questionnaire, 
TIPI personality inventory, and demographic questions.
"""

# Define question options

options_DASS42 = [
    (0, "Did not apply to me at all"),
    (1, "Applied to me to some degree, or some of the time"),
    (2, "Applied to me to a considerable degree, or a good part of time"),
    (3, "Applied to me very much, or most of the time")]
options_TIPI = [
    (1, "Disagree strongly"),
    (2, "Disagree moderately"),
    (3, "Disagree a little"),
    (4, "Neither agree nor disagree"),
    (5, "Agree a little"),
    (6, "Agree moderately"),
    (7, "Agree strongly")]
options_education = [(1.0, "Less than high school"), (2.0, "High school"), (3.0, "University degree"), (4.0, "Graduate degree")]
options_urban = [(0.0, "Rural (country side)"), (1.0, "Suburban"), (2.0, "Urban (town, city)")]
options_gender = [(1, "Male"), (2, "Female"), (3, "Other")]
options_engnat = [(1, "Yes"), (2, "No")]
options_hand = [(1, "Right"), (2, "Left"), (3, "Both"),]
options_orientation = [(1, "Heterosexual"), (2, "Bisexual"), (3, "Homosexual"), (4, "Asexual"), (5, "Other"),]
options_voted = [(1, "Yes"), (2, "No")]
options_married = [(1, "Never married"), (2, "Currently married"), (3, "Previously married")]
options_familysize = [(1, "1"), (2, "2"), (3, "3"), (4, "4"), (5, "5"), (6, "6"), (7, "7"), (8, "8"), (9, "9"), (10, "10"), (11, "11"), (12, "12"), (13, "13")]
options_age_group = [(0.0, "12 or younger"), (1.0, "13-17"), (2.0, "18-24"), (3.0, "25-44"), (4.0, "45-64"), (5.0, "65 or older")]
options_race_group = [("Asian", "Asian"), ("White", "White"), ("Other", "Other")]
options_religion_group = [("Muslim", "Muslim"), ("Christian", "Christian"), ("Other Religion", "Other Religion")]

# Define question fields

fields_DASS42 = [
    ("Q1A", "I found myself getting upset by quite trivial things", options_DASS42),
    ("Q2A", "I was aware of dryness of my mouth", options_DASS42),
    ("Q4A", "I experienced breathing difficulty (e.g. excessively rapid breathing, breathlessness in the absence of physical exertion)", options_DASS42),
    ("Q6A", "I tended to over-react to situations", options_DASS42),
    ("Q7A", "I had a feeling of shakiness (e.g. legs going to give way)", options_DASS42),
    ("Q8A", "I found it difficult to relax", options_DASS42),
    ("Q9A", "I found myself in situations that made me so anxious I was most relieved when they ended", options_DASS42),
    ("Q11A", "I found myself getting upset rather easily", options_DASS42),
    ("Q12A", "I felt that I was using a lot of nervous energy", options_DASS42),
    ("Q14A", "I found myself getting impatient when I was delayed in any way (e.g. elevators, traffic lights, being kept waiting)", options_DASS42),
    ("Q15A", "I had a feeling of faintness", options_DASS42),
    ("Q18A", "I felt that I was rather touchy", options_DASS42),
    ("Q19A", "I perspired noticeably (e.g. hands sweaty) in the absence of high temperatures or physical exertion", options_DASS42),
    ("Q20A", "I felt scared without any good reason", options_DASS42),
    ("Q22A", "I found it hard to wind down", options_DASS42),
    ("Q23A", "I had difficulty in swallowing", options_DASS42),
    ("Q25A", "I was aware of the action of my heart in the absence of physical exertion (e.g. sense of heart rate increase, heart missing a beat)", options_DASS42),
    ("Q27A", "I found that I was very irritable", options_DASS42),
    ("Q28A", "I felt I was close to panic", options_DASS42),
    ("Q29A", "I found it hard to calm down after something upset me", options_DASS42),
    ("Q30A", "I feared that I would be \"thrown\" by some trivial but unfamiliar task", options_DASS42),
    ("Q32A", "I found it difficult to tolerate interruptions to what I was doing", options_DASS42),
    ("Q33A", "I was in a state of nervous tension", options_DASS42),
    ("Q35A", "I was intolerant of anything that kept me from getting on with what I was doing", options_DASS42),
    ("Q36A", "I felt terrified", options_DASS42),
    ("Q39A", "I found myself getting agitated", options_DASS42),
    ("Q40A", "I was worried about situations in which I might panic and make a fool of myself", options_DASS42),
    ("Q41A", "I experienced trembling (e.g. in the hands)", options_DASS42)]
fields_TIPI = [
    ("TIPI1", "I see myself as: Extraverted, enthusiastic", options_TIPI),
    ("TIPI2", "I see myself as: Critical, quarrelsome", options_TIPI),
    ("TIPI3", "I see myself as: Dependable, self-disciplined", options_TIPI),
    ("TIPI4", "I see myself as: Anxious, easily upset", options_TIPI),
    ("TIPI5", "I see myself as: Open to new experiences, complex", options_TIPI),
    ("TIPI6", "I see myself as: Reserved, quiet", options_TIPI),
    ("TIPI7", "I see myself as: Sympathetic, warm", options_TIPI),
    ("TIPI8", "I see myself as: Disorganized, careless", options_TIPI),
    ("TIPI9", "I see myself as: Calm, emotionally stable", options_TIPI),
    ("TIPI10","I see myself as: Conventional, uncreative", options_TIPI)]
fields_demographics = [
    ("education", "How much education have you completed?", options_education),
    ("urban", "What type of area did you live when you were a child?", options_urban),
    ("gender", "What is your gender?", options_gender),
    ("engnat", "Is English your native language?", options_engnat),
    ("hand", "What hand do you use to write with?", options_hand),
    ("orientation", "What is your sexual orientation?", options_orientation),
    ("voted", "Have you voted in a national election in the past year?", options_voted),
    ("married", "What is your marital status?", options_married),
    ("familysize", "Including you, how many children did your mother have?", options_familysize),
    ("age_group", "How many years old are you?", options_age_group),
    ("race_group", "What is your race?", options_race_group),
    ("religion_group", "What is your religion?", options_religion_group)]

