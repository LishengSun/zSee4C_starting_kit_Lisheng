import libscores

score_name = 'mse'
scoring_function = getattr(libscores, score_name)

# this looks for the scoring function defined in libscores such as
#def scoring_function(solution, prediction)
#   ...
#	return score


