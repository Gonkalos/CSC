import gym
import statistics

class PlayerAgent:

    '''
    Constructor
    '''
    def __init__(self, game_steps, nr_games_for_training, score_requirement, game = 'CartPole-v1', log = False):
        self.game_steps = game_steps                       # number of steps (frames) to play for each game
        self.nr_games_for_training = nr_games_for_training # number of games that will be played to build the training set
        self.score_requirement = score_requirement         # minimum score for a "decent" game (for training purposes)
        self.log = log                                     # log info
        self.env = gym.make(game)                          # make the gaming environment
        self.scores = []                                   # list of all scores gotten from played games

    '''
    Play random games
    '''
    def play_random_games(self, nr_random_games):
        for game in range(nr_random_games):
            # reset the environment to the initial state
            self.env.reset()

            for step in range(self.game_steps):
                if self.log: 
                    # render the game
                    self.env.render()
                # get a random action from the list of possible actions
                action = self.env.action_space.sample()
                # execute action 
                observation, reward, done, info = self.env.step(action)
                # if the game is finished
                if done:
                    print('Game %d finished after %d steps' % (game + 1, step + 1))
                    break

    '''
    Build training set
    '''
    def build_training_set(self):
        training_set = []
        score_set = []

        for game in range(self.nr_games_for_training):
            cumulative_game_score = 0
            game_memory = []
            prev_obs = self.env.reset()

            for step in range(self.game_steps):
                # get a random action from the list of possible actions
                action = self.env.action_space.sample()
                # execute action 
                next_obs, reward, done, info = self.env.step(action)
                # store executed action and corresponding observation
                game_memory.append([action, prev_obs])
                cumulative_game_score += reward
                prev_obs = next_obs
                if done:
                    break

            # if the game achieved the minimum score
            if cumulative_game_score >= self.score_requirement:
                # store game score
                score_set.append(cumulative_game_score)
                for play in game_memory:
                    # apply hot-one-encoding to the action
                    if play[0] == 0:
                        one_hot_action = [1, 0]
                    elif play[0] == 1:
                        one_hot_action = [0, 1]
                    # store game memory
                    training_set.append([one_hot_action, play[1]])

        # show some stats
        if score_set:
            print('Number of stored games:', len(training_set))
            print('Average score:', statistics.mean(score_set))

        return training_set

    '''
    Play the game after training
    '''
    def play_game(self):
        # build training set
        training_set = self.build_training_set()
        # use MLP neural network
        mlp = MLP()
        mlp.build()
        mlp.fit(training_set)

        for game in range(self.nr_games_for_training):
            score = 0
            prev_obs = self.env.reset()
            done = False

            while not done:
                if self.log:
                    self.env.render()
                action = mlp.predict(prev_obs)
                next_obs, reward, done, _ = self.env.step(action)
                score += reward
                prev_obs = next_obs

            self.scores.append(score)
            print


player_agent = PlayerAgent(500, 1000, 75)
#player_agent.play_random_games(50)