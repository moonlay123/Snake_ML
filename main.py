import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import snake
from agent import train_dqn
import tensorflow as tf

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "play":
            game = snake.Snake_game()
            game.run()
        else:
            train_dqn()
    else:
        train_dqn()
