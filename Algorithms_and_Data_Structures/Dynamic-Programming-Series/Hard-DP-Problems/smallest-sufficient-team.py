"""
smallest sufficient team
given a list of required skills and a list of people where each person has a set of skills, find the smallest team of people such that all required skills are covered. return the indices of the people in the team.
"""

class Solution:
    def smallestSufficientTeam(self, req_skills, people):
        n = len(req_skills)  # get the number of required skills
        skill_index = {skill: i for i, skill in enumerate(req_skills)}  # map each skill to an index
        dp = {0: []}  # key: bitmask of skills, value: list of people indices
        for i, person in enumerate(people):  # loop through each person
            person_skill = 0  # create a bitmask for this person's skills
            for skill in person:  # loop through each skill of the person
                if skill in skill_index:  # if the skill is required
                    person_skill |= 1 << skill_index[skill]  # set the bit for this skill
            new_dp = dp.copy()  # copy the current dp
            for skill_set, team in dp.items():  # loop through all existing teams
                combined = skill_set | person_skill  # combine skills
                if combined not in new_dp or len(new_dp[combined]) > len(team) + 1:  # if this team is smaller
                    new_dp[combined] = team + [i]  # add this person to the team
            dp = new_dp  # update dp with new teams
        return dp[(1 << n) - 1]  # the answer is the team that covers all skills

    def smallestSufficientTeam(self, req_skills, people):
        n = len(req_skills)
        skill_index = {skill: i for i, skill in enumerate(req_skills)}  # map each skill to an index
        dp = {0: []}  # Key: bitmask of skills, Value: list of people indices
        
        for i, person in enumerate(people):
            # create a bitmask for this person's skills
            person_skill = 0
            for skill in person:
                if skill in skill_index:
                    person_skill |= 1 << skill_index[skill]
            # try to add this person to all existing teams
            new_dp = dp.copy()
            for skill_set, team in dp.items():
                combined = skill_set | person_skill  # combine skills
                if combined not in new_dp or len(new_dp[combined]) > len(team) + 1:
                    new_dp[combined] = team + [i]  # add this person to the team
            dp = new_dp  # update dp with new teams
        # the answer is the team that covers all skills (all bits set)
        return dp[(1 << n) - 1] 