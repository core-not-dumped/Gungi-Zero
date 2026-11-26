"""대국 loop. 어떤 두 AI 든 한 판을 두고, 결과는 선수·후수로 적힌다."""

from ai.episode import play_episode
from games.base import Outcome, Player
from games.tictactoe import TicTacToe, TicTacToeMove, TicTacToePosition


class PreferringAI:
    """적힌 순서에서 처음으로 합법인 수를 둔다. 자기가 받은 국면의 차례를 남긴다."""

    def __init__(self, game: TicTacToe, preference: str) -> None:
        self._game = game
        self._preference = tuple(game.parse_move(text) for text in preference.split())
        self.asked: list[Player] = []

    def choose(self, position: TicTacToePosition) -> TicTacToeMove:
        self.asked.append(position.player)
        legal = self._game.legal_moves(position)
        return next(move for move in self._preference if move in legal)


# 아랫줄을 채우는 AI 는 가운뎃줄과 a3 부터 두는 AI 를 선후와 무관하게 이긴다.
WINNER = "a1 b1 c1"
LOSER = "a2 b2 a3 c3"


def rendered(game: TicTacToe, moves: tuple[TicTacToeMove, ...]) -> str:
    return " ".join(game.render_move(move) for move in moves)


def test_the_two_ais_take_turns_starting_with_the_first() -> None:
    game = TicTacToe()
    first, second = PreferringAI(game, WINNER), PreferringAI(game, LOSER)
    episode = play_episode(game, first, second)
    assert rendered(game, episode.moves) == "a1 a2 b1 b2 c1"
    assert first.asked == [Player.FIRST] * 3
    assert second.asked == [Player.SECOND] * 2


def test_the_episode_ends_where_the_game_ends() -> None:
    game = TicTacToe()
    episode = play_episode(game, PreferringAI(game, WINNER), PreferringAI(game, LOSER))
    assert episode.outcome is Outcome.FIRST_WINS
    assert game.outcome(episode.final_position) is episode.outcome
    assert episode.final_position.ply == len(episode.moves)


def test_swapping_the_ais_records_the_result_by_seat() -> None:
    game = TicTacToe()
    episode = play_episode(game, PreferringAI(game, LOSER), PreferringAI(game, WINNER))
    assert rendered(game, episode.moves) == "a2 a1 b2 b1 a3 c1"
    assert episode.outcome is Outcome.SECOND_WINS
