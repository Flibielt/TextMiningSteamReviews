games = []
training_dataset = []
testing_dataset = []
game_tags = set()


class Game:
    def __init__(self):
        self.title = ""
        self.developer = ""
        self.publisher = ""
        self.tags = []
        self.overview = ""


class UserReview:
    def __init__(self):
        self.id = 0
        self.title = ""
        self.year = 0
        self.user_review = ""
        self.suggested = False
