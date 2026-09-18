import math

from world.game_state import GameState


def base_evaluation_function(state: GameState) -> float:
    """
    Retorna la evaluación base entregada para desarrollar el punto 4.

    Esta función no forma parte del código que debe modificar el estudiante y
    permite probar Minimax antes de desarrollar la heurística del punto 5.
    """
    if state.is_win():
        return 1000.0
    if state.is_lose():
        return -1000.0
    return float(state.get_score())


def evaluation_function(state: GameState) -> float:
    """
    Evalúa un estado desde la perspectiva del defensor MAX.

    Debe conservar las utilidades terminales de la evaluación base y diseñar
    una valoración no trivial para estados de corte. Minimax y alfa-beta usan
    esta misma función al comparar sus decisiones en el punto 5.

    Tips:
    - Los estados terminales ya se resuelven antes del bloque TODO; diseñe allí
      únicamente la valoración de estados no terminales.
    - Consulte state.defender_position, state.intruder_position,
      state.pending_terminals, state.get_score() y state.get_legal_actions(0).
    - state.layout.distance(start, goal) calcula y almacena en caché la distancia
      real por el mapa respetando los muros.
    - Maneje conjuntos vacíos y distancias infinitas, y mantenga todo estado no
      terminal estrictamente entre -1000 y +1000.
    """
    if state.is_win() or state.is_lose():
        return base_evaluation_function(state)

    distances = [
        state.layout.distance(state.defender_position, terminal)
        for terminal in sorted(state.pending_terminals)
    ]
    reachable = [distance for distance in distances if math.isfinite(distance)]
    nearest = min(reachable, default=0)
    unreachable = len(distances) - len(reachable)
    separation = state.layout.distance(state.defender_position, state.intruder_position)
    # La bonificación de seguridad se satura para no premiar huir indefinidamente.
    safety = min(separation, 6)
    immediate_risk = 1 if separation <= 1 else 0
    mobility = sum(action != "Stop" for action in state.get_legal_actions(0))

    value = (
        state.get_score()
        - 80 * len(state.pending_terminals)
        - 8 * nearest
        - 200 * unreachable
        + 6 * safety
        - 180 * immediate_risk
        + 3 * mobility
    )
    # Incluso puntajes acumulados extremos deben quedar debajo de la utilidad terminal.
    return max(-999.0, min(999.0, float(value)))
