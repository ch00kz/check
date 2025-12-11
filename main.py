from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta
from enum import StrEnum
from typing import Literal

import click
import requests
from pydantic import BaseModel, Field, HttpUrl
from rich.console import Console
from rich.progress import track
from rich.table import Table


# =============================================================================
# Type Aliases
# =============================================================================

TimeClass = Literal["bullet", "blitz", "rapid", "daily"]
PlayerResult = Literal["win", "loss", "draw"]
AggregationMode = Literal["daily", "weekly"]

type DailySummaryMap = dict[date, DailySummary]
type WeeklySummaryMap = dict[date, WeeklySummary]
type TimeClassDailySummaries = dict[str, DailySummaryMap]
type TimeClassWeeklySummaries = dict[str, WeeklySummaryMap]
type MonthSpec = tuple[int, int]  # (year, month)


# =============================================================================
# Enums
# =============================================================================


class GameResult(StrEnum):
    """Chess.com game result values."""

    WIN = "win"
    CHECKMATED = "checkmated"
    TIMEOUT = "timeout"
    RESIGNED = "resigned"
    STALEMATE = "stalemate"
    INSUFFICIENT = "insufficient"
    FIFTY_MOVE = "50move"
    REPETITION = "repetition"
    AGREED = "agreed"
    TIMEVSINSUFFICIENT = "timevsinsufficient"
    ABANDONED = "abandoned"
    KINGOFTHEHILL = "kingofthehill"
    THREECHECK = "threecheck"
    BUGHOUSEPARTNERLOSE = "bughousepartnerlose"

    @property
    def is_win(self) -> bool:
        return self == GameResult.WIN


# =============================================================================
# API Models
# =============================================================================


class Accuracies(BaseModel):
    """Game accuracy percentages for both players."""

    white: float
    black: float


class Player(BaseModel):
    """Player information within a game."""

    rating: int
    result: GameResult
    id: HttpUrl = Field(alias="@id")
    username: str
    uuid: str

    @property
    def won(self) -> bool:
        return self.result.is_win


class ChessGame(BaseModel):
    """A single chess game from Chess.com."""

    url: HttpUrl
    pgn: str
    time_control: str
    end_time: int
    rated: bool
    accuracies: Accuracies | None = None
    tcn: str
    uuid: str
    initial_setup: str
    fen: str
    time_class: str
    rules: str
    white: Player
    black: Player
    eco: HttpUrl | None = None

    @property
    def end_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.end_time)

    @property
    def winner(self) -> str | None:
        """Username of the winner, or None if draw."""
        if self.white.won:
            return self.white.username
        if self.black.won:
            return self.black.username
        return None

    @property
    def is_draw(self) -> bool:
        return self.winner is None

    def get_player_result(self, username: str) -> PlayerResult:
        """Get result for a specific player: 'win', 'loss', or 'draw'."""
        username_lower = username.lower()
        if self.winner:
            return "win" if self.winner.lower() == username_lower else "loss"
        return "draw"

    def get_player_data(self, username: str) -> Player | None:
        """Get Player object for specified username."""
        username_lower = username.lower()
        if self.white.username.lower() == username_lower:
            return self.white
        if self.black.username.lower() == username_lower:
            return self.black
        return None


class GamesResponse(BaseModel):
    """API response containing a list of games."""

    games: list[ChessGame]


# =============================================================================
# Summary Models
# =============================================================================


class SummaryModel(BaseModel):
    """Base model for game summaries with win/loss/draw tracking."""

    wins: int = 0
    losses: int = 0
    draws: int = 0
    starting_rating: int | None = None
    ending_rating: int | None = None

    @property
    def games_played(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_percentage(self) -> float:
        if self.games_played > 0:
            return (self.wins / self.games_played) * 100
        return 0.0

    @property
    def win_percentage_str(self) -> str:
        return f"{self.win_percentage:.1f}%"

    @property
    def rating_change(self) -> int | None:
        """Calculate rating change, or None if ratings unavailable."""
        if self.ending_rating is not None and self.starting_rating is not None:
            return self.ending_rating - self.starting_rating
        return None

    @property
    def rating_change_str(self) -> str:
        change = self.rating_change
        if change is None:
            return "[dim]N/A[/]"
        return format_rating_change(change)


class DailySummary(SummaryModel):
    """Summary of games for a single day."""

    date: date
    games: list[ChessGame] = Field(default_factory=list)

    def update_from_game(self, game: ChessGame, username: str) -> None:
        """Update summary based on a game."""
        self.games.append(game)

        player = game.get_player_data(username)
        if player is None:
            return

        result = game.get_player_result(username)

        if self.starting_rating is None:
            self.starting_rating = player.rating
        self.ending_rating = player.rating

        match result:
            case "win":
                self.wins += 1
            case "loss":
                self.losses += 1
            case "draw":
                self.draws += 1


class WeeklySummary(SummaryModel):
    """Summary of games for a week (Monday-Sunday)."""

    week_start: date
    week_end: date
    daily_summaries: dict[date, DailySummary] = Field(default_factory=dict)

    def calculate_metrics(self) -> None:
        """Calculate aggregated metrics from daily summaries."""
        if not self.daily_summaries:
            return

        sorted_dates = sorted(self.daily_summaries.keys())

        # Aggregate totals
        self.wins = sum(d.wins for d in self.daily_summaries.values())
        self.losses = sum(d.losses for d in self.daily_summaries.values())
        self.draws = sum(d.draws for d in self.daily_summaries.values())

        # Get first and last ratings
        first_day = self.daily_summaries[sorted_dates[0]]
        last_day = self.daily_summaries[sorted_dates[-1]]
        self.starting_rating = first_day.starting_rating
        self.ending_rating = last_day.ending_rating


# =============================================================================
# API Client
# =============================================================================


class ChessComClient:
    """Client for the Chess.com public API."""

    BASE_URL = "https://api.chess.com/pub"
    REQUEST_DELAY = 0.1  # seconds between requests

    def __init__(self, user_agent: str = "Check/1.0 (check-stats)") -> None:
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent

    def fetch_month_games(
        self, username: str, year: int, month: int
    ) -> list[ChessGame]:
        """Fetch all games for a user in a specific month."""
        url = f"{self.BASE_URL}/player/{username}/games/{year}/{month:02d}"
        response = self.session.get(url)

        if response.status_code != 200:
            console = Console()
            console.print(f"[red]Error {response.status_code} fetching {url}[/]")
            return []

        return GamesResponse(**response.json()).games

    def fetch_games_for_months(
        self, username: str, months: list[MonthSpec]
    ) -> list[ChessGame]:
        """Fetch games for multiple months with progress tracking."""
        import time

        all_games: list[ChessGame] = []

        for year, month in track(months, description=f"Downloading {username}'s games"):
            month_games = self.fetch_month_games(username, year, month)
            all_games.extend(month_games)
            time.sleep(self.REQUEST_DELAY)

        return all_games


# =============================================================================
# Aggregation Functions
# =============================================================================


def get_week_bounds(date_obj: date) -> tuple[date, date]:
    """Get Monday and Sunday for the week containing the given date."""
    days_since_monday = date_obj.weekday()
    week_start = date_obj - timedelta(days=days_since_monday)
    week_end = week_start + timedelta(days=6)
    return week_start, week_end


def _build_daily_summaries(
    games: list[ChessGame], username: str
) -> dict[str, DailySummaryMap]:
    """Build daily summaries grouped by time class."""
    # Group games by time class
    games_by_time_class: dict[str, list[ChessGame]] = defaultdict(list)

    for game in games:
        if game.rated and game.get_player_data(username) is not None:
            games_by_time_class[game.time_class].append(game)

    # Sort each group by end time
    for game_list in games_by_time_class.values():
        game_list.sort(key=lambda g: g.end_time)

    # Build daily summaries for each time class
    result: dict[str, DailySummaryMap] = {}

    for time_class, game_list in games_by_time_class.items():
        daily_summaries: DailySummaryMap = {}
        previous_game: ChessGame | None = None

        for game in game_list:
            game_date = game.end_datetime.date()

            if game_date not in daily_summaries:
                # Initialize with previous day's ending rating if available
                initial_rating: int | None = None
                if previous_game:
                    prev_player = previous_game.get_player_data(username)
                    if prev_player:
                        initial_rating = prev_player.rating

                daily_summaries[game_date] = DailySummary(
                    date=game_date, starting_rating=initial_rating
                )

            daily_summaries[game_date].update_from_game(game, username)
            previous_game = game

        result[time_class] = daily_summaries

    return result


def aggregate_daily(
    all_games: list[ChessGame], username: str
) -> TimeClassDailySummaries:
    """Aggregate games into daily summaries by time class."""
    return _build_daily_summaries(all_games, username)


def aggregate_weekly(
    all_games: list[ChessGame], username: str
) -> TimeClassWeeklySummaries:
    """Aggregate games into weekly summaries by time class."""
    daily_by_time_class = _build_daily_summaries(all_games, username)

    result: TimeClassWeeklySummaries = {}

    for time_class, daily_summaries in daily_by_time_class.items():
        weekly_summaries: WeeklySummaryMap = {}

        for day, daily_summary in daily_summaries.items():
            week_start, week_end = get_week_bounds(day)

            if week_start not in weekly_summaries:
                weekly_summaries[week_start] = WeeklySummary(
                    week_start=week_start, week_end=week_end
                )

            weekly_summaries[week_start].daily_summaries[day] = daily_summary

        # Calculate metrics for each week
        for weekly in weekly_summaries.values():
            weekly.calculate_metrics()

        result[time_class] = weekly_summaries

    return result


# =============================================================================
# Display Functions
# =============================================================================


def format_rating_change(change: int) -> str:
    """Format a rating change with color coding."""
    if change > 0:
        return f"[bold green]+{change}[/]"
    if change < 0:
        return f"[bold red]{change}[/]"
    return "[white]0[/]"


def _create_summary_table(title: str) -> Table:
    """Create a table with standard columns for summaries."""
    table = Table(title=title)
    table.add_column("Period", justify="center", no_wrap=True)
    table.add_column("Games", justify="center")
    table.add_column("Wins", justify="right", style="bold green")
    table.add_column("Losses", justify="right", style="bold red")
    table.add_column("Draws", justify="right")
    table.add_column("Win %", justify="right")
    table.add_column("Start Rating", justify="right")
    table.add_column("End Rating", justify="right")
    table.add_column("+/-", justify="right")
    return table


def _add_totals_row(
    table: Table,
    total_games: int,
    total_wins: int,
    total_losses: int,
    total_draws: int,
    start_rating: int | None,
    end_rating: int | None,
) -> None:
    """Add a totals row to a summary table."""
    win_pct = (total_wins / total_games * 100) if total_games > 0 else 0.0

    rating_change_str: str
    if start_rating is not None and end_rating is not None:
        rating_change_str = format_rating_change(end_rating - start_rating)
    else:
        rating_change_str = "[dim]N/A[/]"

    table.add_section()
    table.add_row(
        "Totals",
        str(total_games),
        str(total_wins),
        str(total_losses),
        str(total_draws),
        f"{win_pct:.1f}%",
        str(start_rating) if start_rating else "N/A",
        str(end_rating) if end_rating else "N/A",
        rating_change_str,
    )


def print_daily_summaries(
    summaries: DailySummaryMap, username: str, time_class: str
) -> None:
    """Print a table of daily summaries."""
    if not summaries:
        Console().print(f"[yellow]No {time_class} games found for @{username}[/]")
        return

    table = _create_summary_table(f'"{time_class.title()}" summary for @{username}')

    sorted_dates = sorted(summaries.keys())
    total_games = 0
    total_wins = 0
    total_losses = 0
    total_draws = 0
    start_rating = summaries[sorted_dates[0]].starting_rating
    end_rating: int | None = None

    for day in sorted_dates:
        summary = summaries[day]
        total_games += summary.games_played
        total_wins += summary.wins
        total_losses += summary.losses
        total_draws += summary.draws
        end_rating = summary.ending_rating

        table.add_row(
            summary.date.strftime("%Y-%m-%d"),
            str(summary.games_played),
            str(summary.wins),
            str(summary.losses),
            str(summary.draws),
            summary.win_percentage_str,
            str(summary.starting_rating) if summary.starting_rating else "N/A",
            str(summary.ending_rating) if summary.ending_rating else "N/A",
            summary.rating_change_str,
        )

    _add_totals_row(
        table,
        total_games,
        total_wins,
        total_losses,
        total_draws,
        start_rating,
        end_rating,
    )

    Console().print(table)


def print_weekly_summaries(
    summaries: WeeklySummaryMap, username: str, time_class: str
) -> None:
    """Print a table of weekly summaries."""
    if not summaries:
        Console().print(f"[yellow]No {time_class} games found for @{username}[/]")
        return

    table = _create_summary_table(f'"{time_class.title()}" summary for @{username}')

    sorted_weeks = sorted(summaries.keys())
    total_games = 0
    total_wins = 0
    total_losses = 0
    total_draws = 0
    start_rating = summaries[sorted_weeks[0]].starting_rating
    end_rating: int | None = None

    for week_start in sorted_weeks:
        summary = summaries[week_start]
        total_games += summary.games_played
        total_wins += summary.wins
        total_losses += summary.losses
        total_draws += summary.draws
        end_rating = summary.ending_rating

        period = f"{summary.week_start:%Y-%m-%d} - {summary.week_end:%Y-%m-%d}"
        table.add_row(
            period,
            str(summary.games_played),
            str(summary.wins),
            str(summary.losses),
            str(summary.draws),
            summary.win_percentage_str,
            str(summary.starting_rating) if summary.starting_rating else "N/A",
            str(summary.ending_rating) if summary.ending_rating else "N/A",
            summary.rating_change_str,
        )

    _add_totals_row(
        table,
        total_games,
        total_wins,
        total_losses,
        total_draws,
        start_rating,
        end_rating,
    )

    Console().print(table)


# =============================================================================
# Month Parsing
# =============================================================================


def parse_months(months_str: str) -> list[MonthSpec]:
    """
    Parse month specification string into list of (year, month) tuples.

    Examples:
        "2024/12" -> [(2024, 12)]
        "2024/11,2024/12" -> [(2024, 11), (2024, 12)]
        "2024/10-12" -> [(2024, 10), (2024, 11), (2024, 12)]
        "2024/11-2025/1" -> [(2024, 11), (2024, 12), (2025, 1)]
    """
    months: list[MonthSpec] = []

    for part in months_str.split(","):
        part = part.strip()

        if "-" in part and "/" in part:
            months.extend(_parse_month_range(part))
        elif "/" in part:
            year, month = map(int, part.split("/"))
            months.append((year, month))

    return months


def _parse_month_range(part: str) -> list[MonthSpec]:
    """Parse a month range like '2024/10-12' or '2024/11-2025/1'."""
    months: list[MonthSpec] = []

    if part.count("/") == 1:
        # Single year range: "2024/10-12"
        year_str, month_range = part.split("/")
        year = int(year_str)
        start_month, end_month = map(int, month_range.split("-"))
        for month in range(start_month, end_month + 1):
            if 1 <= month <= 12:
                months.append((year, month))
    else:
        # Cross-year range: "2024/11-2025/1"
        start, end = part.split("-")
        start_year, start_month = map(int, start.split("/"))
        end_year, end_month = map(int, end.split("/"))

        current_year, current_month = start_year, start_month
        while (current_year, current_month) <= (end_year, end_month):
            months.append((current_year, current_month))
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1

    return months


# =============================================================================
# Main Analysis
# =============================================================================


def analyze_player_games(
    username: str,
    months: list[MonthSpec],
    mode: AggregationMode = "daily",
) -> None:
    """Fetch and analyze games for specified months."""
    console = Console()
    client = ChessComClient()

    all_games = client.fetch_games_for_months(username, months)

    if not all_games:
        console.print("[red]No games found![/]")
        return

    rated_count = sum(1 for g in all_games if g.rated and g.get_player_data(username))
    console.print(f"[cyan]{rated_count}/{len(all_games)} games are rated[/]")

    if mode == "weekly":
        weekly_summaries = aggregate_weekly(all_games, username)
        for time_class, week_data in weekly_summaries.items():
            print_weekly_summaries(week_data, username, time_class)
    else:
        daily_summaries = aggregate_daily(all_games, username)
        for time_class, day_data in daily_summaries.items():
            print_daily_summaries(day_data, username, time_class)


# =============================================================================
# CLI
# =============================================================================


@click.command()
@click.option(
    "--username",
    "-u",
    prompt="Chess.com username",
    help="Chess.com username",
)
@click.option(
    "--months",
    "-m",
    prompt='Time span (e.g., "2024/12", "2024/11,2024/12", "2024/10-12", "2024/11-2025/1")',
    help="Months to analyze",
)
@click.option(
    "--aggregation-mode",
    "-a",
    prompt="Aggregation mode",
    type=click.Choice(["daily", "weekly"]),
    default="daily",
    help="Aggregation mode: daily or weekly",
)
def summary(username: str, months: str, aggregation_mode: AggregationMode) -> None:
    """Analyze Chess.com games with daily or weekly aggregation."""
    console = Console()

    if not months:
        now = datetime.now()
        months_list = [(now.year, now.month)]
        console.print(
            f"[yellow]No months specified, using current month: {now.year}/{now.month}[/]"
        )
    else:
        months_list = parse_months(months)
        formatted = ", ".join(f"{y}/{m}" for y, m in months_list)
        console.print(f"[cyan]Analyzing months: {formatted}[/]")

    analyze_player_games(username, months_list, aggregation_mode)


if __name__ == "__main__":
    summary()
