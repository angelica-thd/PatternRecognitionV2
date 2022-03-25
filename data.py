import sqlite3
from numpy import array

#Fetching data
def fetchData():
    conn = sqlite3.connect('database.sqlite')
    c = conn.cursor()

    c.execute('SELECT B365H, B365D, B365A, home_team_goal, away_team_goal FROM Match WHERE B365H NOT NULL')
    B365 = array(c.fetchall())

    c.execute('SELECT BWH, BWD, BWA, home_team_goal, away_team_goal FROM Match WHERE BWH NOT NULL')
    BW = array(c.fetchall())

    c.execute('SELECT IWH, IWD, IWA, home_team_goal, away_team_goal FROM Match WHERE IWH NOT NULL')
    IW = array(c.fetchall())

    c.execute('''SELECT LBH, LBD, LBA, home_team_goal, away_team_goal FROM Match WHERE LBH NOT NULL''')
    LB = array(c.fetchall())

    c.execute('''SELECT h.buildUpPlaySpeed, h.buildUpPlayPassing, h.chanceCreationPassing, h.chanceCreationCrossing,
                        h.chanceCreationShooting, h.defencePressure, h.defenceAggression, h.defenceTeamWidth,
                        a.buildUpPlaySpeed, a.buildUpPlayPassing, a.chanceCreationPassing, a.chanceCreationCrossing,
                        a.chanceCreationShooting, a.defencePressure, a.defenceAggression, a.defenceTeamWidth, B365H, B365D, B365A, BWH, BWD, BWA, 
                        IWH, IWD, IWA, LBH, LBD, LBA, home_team_goal, away_team_goal
                            FROM Match m INNER JOIN Team_Attributes h ON m.home_team_api_id=h.team_api_id AND m.date = h.date 
                                            INNER JOIN Team_Attributes a ON m.away_team_api_id=a.team_api_id AND m.date = a.date 
                                            WHERE B365H NOT NULL AND BWH NOT NULL AND IWH NOT NULL AND LBH NOT NULL''')
    TeamAttributesData = array(c.fetchall())

    return (B365, BW, IW, LB, TeamAttributesData)