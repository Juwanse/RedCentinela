from abc import ABC, abstractmethod

from algorithms.evaluation import evaluation_function
from world.game_state import GameState


class MultiAgentSearchAgent(ABC):
    """Clase base para los agentes de búsqueda adversaria."""

    def __init__(self, depth: int | str = 2) -> None:
        self.depth = int(depth)
        if self.depth < 1:
            raise ValueError("La profundidad debe ser al menos 1 ply")
        self.nodes_evaluated = 0

    @abstractmethod
    def get_action(self, state: GameState) -> str | None:
        raise NotImplementedError


class MinimaxAgent(MultiAgentSearchAgent):
    """Agente Minimax para el defensor MAX frente al intruso MIN."""

    def get_action(self, state: GameState) -> str | None:
        """
        Retorna la acción del defensor con mayor valor Minimax.

        El defensor es MAX (agente 0), el intruso es MIN (agente 1)
        y cada acción consume un ply.

        Se conserva la primera acción en caso de empate y se cuenta
        cada estado procesado una sola vez en self.nodes_evaluated.
        """
        self.nodes_evaluated = 0

        def search(
            node: GameState,
            agent_index: int,
            remaining_depth: int,
        ):
            self.nodes_evaluated += 1
            if (
                node.is_win()
                or node.is_lose()
                or remaining_depth == 0
            ):
                return evaluation_function(node), None

            actions = node.get_legal_actions(agent_index)
            if not actions:
                return evaluation_function(node), None
            next_agent = (
                agent_index + 1
            ) % node.get_num_agents()
            if agent_index == 0:
                best_value = float("-inf")
                best_action = actions[0]

                for action in actions:
                    successor = node.generate_successor(
                        agent_index,
                        action,
                    )
                    value, _ = search(
                        successor,
                        next_agent,
                        remaining_depth - 1,
                    )
                    if value > best_value:
                        best_value = value
                        best_action = action

                return best_value, best_action
            else:
                best_value = float("inf")
                best_action = actions[0]

                for action in actions:
                    successor = node.generate_successor(
                        agent_index,
                        action,
                    )
                    value, _ = search(
                        successor,
                        next_agent,
                        remaining_depth - 1,
                    )
                    if value < best_value:
                        best_value = value
                        best_action = action

                return best_value, best_action

        _, action = search(
            state,
            0,
            self.depth,
        )

        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """Agente Minimax que evita explorar ramas mediante poda alfa-beta."""

    def get_action(self, state: GameState) -> str | None:
        """
        Retorna la acción de Minimax aplicando poda alfa-beta.

        Debe usar la misma profundidad, orden de acciones y función de
        evaluación que Minimax.

        Tips:
        - Conserve la misma estructura y casos base de MinimaxAgent.
        - Inicie alpha en -infinito y beta en +infinito, y páselos en las
          llamadas recursivas.
        - En MAX actualice alpha y corte si valor >= beta; en MIN actualice beta
          y corte si valor <= alpha.
        """
        self.nodes_evaluated = 0

        def search(node, agent_index, remaining, alpha, beta):
            self.nodes_evaluated += 1
            if node.is_win() or node.is_lose() or remaining == 0:
                return evaluation_function(node), None
            actions = node.get_legal_actions(agent_index)
            if not actions:
                return evaluation_function(node), None

            maximizing = agent_index == 0
            best_value = float("-inf") if maximizing else float("inf")
            best_action = actions[0]
            next_agent = (agent_index + 1) % node.get_num_agents()
            for action in actions:
                value, _ = search(
                    node.generate_successor(agent_index, action),
                    next_agent, remaining - 1, alpha, beta,
                )
                # Las comparaciones estrictas conservan la primera acción empatada.
                if maximizing:
                    if value > best_value:
                        best_value, best_action = value, action
                    alpha = max(alpha, best_value)
                else:
                    if value < best_value:
                        best_value, best_action = value, action
                    beta = min(beta, best_value)
                if alpha >= beta:
                    break
            return best_value, best_action

        _, action = search(state, 0, self.depth, float("-inf"), float("inf"))
        return action
