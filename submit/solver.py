from model import ReviewClassifierModel

'''
USAGE: SOLVER
'''

class ClassifyReviewSolver:
    def __init__(self):
        self.solver = ReviewClassifierModel()
        self.solver.setup_classifier()

    def solve(self, text):
        #out = PreProcess(text)
        out = self.solver(text)
        return out

