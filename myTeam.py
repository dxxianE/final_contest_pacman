# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from itertools import combinations
import random
import util as util

from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        #añadire nuevas condiciones para decidir si el agente debe de moverse hacia la base o continuar, esto
        #se basara en la cantidad de domida que quede y el puntuaje. 

        #if food_left <= 2:

        our_food_left = len(self.get_food_you_are_defending(game_state).as_list())
        score = self.get_score(game_state)

        if food_left < our_food_left and score <= 0:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):

        #Datos relevante del estado actual
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        food_left = len(self.get_food(game_state).as_list())
        our_food_left = len(self.get_food_you_are_defending(game_state).as_list())
        score = self.get_score(game_state)
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        no_invaders = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        if len(invaders) > 0 or score > 0 or our_food_left > food_left:
            #características para el modo defensa/ofensa
            features['on_defense'] = 1 if my_state.is_pacman else 0

            
            if action == Directions.STOP:
                features['stop'] = 1

             #calculamos la distancia entre miembros del equipo
            team_positions = [game_state.get_agent_state(i).get_position() for i in self.get_team(game_state)]
            dist_between_members = float('inf')

            #distancia mínima entre los miembros del equipo utilizando combinaciones
            if len(team_positions) > 1:
                dist_between_members = min(self.get_maze_distance(*positions) for positions in combinations(team_positions, 2))
            
            #dist_members en la distancia ajustada
            #solo se aplica si estamos en modo defensa (score > 0)
            features['dist_members'] = max(0, dist_between_members - 7) if score > 0 else 0

            #distancia a los enemigos no invasores
            no_invader_dists = [self.get_maze_distance(my_pos, a.get_position()) for a in no_invaders]
            features['no_invader_distance'] = min(no_invader_dists) if no_invaders else 0

            # num de invasores y su distancia más cercana
            features['num_invaders'] = len(invaders)

            if invaders:
                invader_dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
                features['invader_distance'] = min(invader_dists)
            else:
                #dists al inicio si no hay invasores
                features['avoid_start'] = self.get_maze_distance(my_pos, self.start)

            #detectamos si la acción es detenerse o dar la vuelta
            if action == Directions.STOP:
                features['stop'] = 1
            rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
            if action == rev:
                features['reverse'] = 1
        else:
            #características para el modo ataque
            food_list = self.get_food(successor).as_list()
            features['successor_score'] = -len(food_list)

            #distancia a los enemigos no invasores
            no_inv_dists = [self.get_maze_distance(my_pos, a.get_position()) for a in no_invaders]
            features['no_invader_distance'] = 0 if not no_invaders else 1000 - min(no_inv_dists)

            #detectamos si somos un agente de ataque
            features['attack'] = 1 if not my_state.is_pacman else 0

            #distancia al alimento más cercano
            if food_list:
                features['distance_to_food'] = min(self.get_maze_distance(my_pos, food) for food in food_list)

            #detectamos si hemos marcado o misseado puntos
            features['scores'] = 1 if score > 0 else -1

        return features
    
    def get_weights(self, game_state, action):

        successor = self.get_successor(game_state, action)
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        score = self.get_score(game_state)

        if len(invaders) > 0:
            return {'num_invaders': -1000000, 'invader_distance': -5000, 'stop': -6000}
        elif score > 0:
            return {'on_defense': 100000, 'stop': -1000, 'reverse': -2, 'no_invader_distance': -500, 'avoid_start': 100, 'dist_members': 100}
        else:
            return {'successor_score': 10, 'distance_to_food': -1, 'stop': -300, 'scores': 1000, 'no_invader_distance': -100, 'attack': -500}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """
    def get_features(self, game_state, action):

        features = util.Counter()
        successor = self.get_successor(game_state, action)

        #obtenemos el estado del agente actual después de tomar la acción
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: 
            features['on_defense'] = 0
        
        #posición de los miembros del equipo
        team = self.get_team(game_state)
        agent_1 = game_state.get_agent_state(team[0])
        pos_ag_1 = agent_1.get_position()
        agent_2 = game_state.get_agent_state(team[1])
        pos_ag_2 = agent_2.get_position()

        #distancia entre los miembros del equipo
        dist_between_members = self.get_maze_distance(pos_ag_1, pos_ag_2)
        
        #si el equipo ha anotado puntos, ajusta la característica de distancia entre miembros
        score = self.get_score(game_state)
        features['dist_members'] = 0
        if score > 0:
            dist_m = 7
            if dist_between_members < dist_m:
                features['dist_members'] = dist_between_members-dist_m
            else:
                features['dist_members'] = 0

        #la información de los enemigos en el sucesor 8distancia)
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        #filtra los enemigos que son Pac-Man (invasores)
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        #numero de invasores y su distancia más cercana
        features['num_invaders'] = len(invaders)

        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)
        else:
            dist_start = self.get_maze_distance(my_pos, self.start)
            features['avoid_start'] = dist_start

        #calcula la distancia a los enemigos que no son Pac-Man (no invasores)
        no_invaders = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        no_inv_dists = [self.get_maze_distance(my_pos, a.get_position()) for a in no_invaders]
        
        if len(no_invaders) == 0:
            features['no_invader_distance'] = 0
        else:
            features['no_invader_distance'] = min(no_inv_dists)

        #estimamos si la acción es detenerse o dar la vuelta
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features


    def get_weights(self, game_state, action):
        successor = self.get_successor(game_state, action)
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        
        if len(invaders) > 0:
            return {'num_invaders': -1000000, 'invader_distance': -5000}
        else:
            return {'on_defense': 100000, 'stop': -1000, 'reverse': -2, 'no_invader_distance': -500, 'avoid_start': 100, 'dist_members': 500}