from model import ReviewClassifierModel

'''
USAGE: SOLVER
'''

class ClassifyReviewSolver:
    def __init__(self):
        self.solver = ReviewClassifierModel()

    def solve(self, text):
        #out = PreProcess(text)
        self.solver.setup_classifier()
        out = self.solver(text)
        return out

