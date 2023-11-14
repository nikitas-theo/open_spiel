// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/yacht/yacht.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace yacht {
namespace {

const std::vector<std::pair<Action, double>> kChanceOutcomes = {
    std::pair<Action, double>(1, 1.0 / 6),
    std::pair<Action, double>(2, 1.0 / 6),
    std::pair<Action, double>(3, 1.0 / 6),
    std::pair<Action, double>(4, 1.0 / 6),
    std::pair<Action, double>(5, 1.0 / 6),
    std::pair<Action, double>(6, 1.0 / 6),
};

const std::vector<int> kChanceOutcomeValues = {1, 2, 3, 4, 5, 6};

constexpr int kLowestDieRoll = 1;
constexpr int kHighestDieRoll = 6;
constexpr int kInitialTurn = -1;

// Facts about the game
const GameType kGameType{/*short_name=*/"yacht",
                         /*long_name=*/"Yacht",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kPerfectInformation,
                         GameType::Utility::kZeroSum,
                         GameType::RewardModel::kTerminal,
                         /*min_num_players=*/2,
                         /*max_num_players=*/2,
                         /*provides_information_state_string=*/false,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true};

static std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new YachtGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);
}  // namespace

std::string CurPlayerToString(Player cur_player) {
  switch (cur_player) {
    case 1:
      return "Player 1";
    case 2:
      return "Player 2";
    case kChancePlayerId:
      return "*";
    case kTerminalPlayerId:
      return "T";
    default:
      SpielFatalError(absl::StrCat("Unrecognized player id: ", cur_player));
  }
}

std::string PositionToStringHumanReadable(int pos) { return "Pos"; }

std::string YachtState::ActionToString(Player player, Action move_id) const {
  if (player == kChancePlayerId) {
    return absl::StrCat("chance outcome ", move_id,
                        " (roll: ", kChanceOutcomeValues[move_id - 1], ")");
  } else {
    if (move_id >= kLowestDieRoll && move_id <= kHighestDieRoll) {
      return absl::StrCat("Player ", player, ": chose to re-roll die ",
                          move_id);
    } else if (move_id == kPass) {
      if (dice_to_reroll_.empty()) {
        return absl::StrCat("Player ", player, ": chose to reroll no dice.");
      } else {
        std::string reroll_dice = "";
        for (int i = 0; i < dice_to_reroll_.size() - 1; ++i) {
          reroll_dice += DiceToString(dice_to_reroll_[i]) + ", ";
        }
        reroll_dice +=
            DiceToString(dice_to_reroll_[dice_to_reroll_.size() - 1]);
        return absl::StrCat("Player ", player, ": chose to roll dice ",
                            reroll_dice);
      }
    } else {
      return absl::StrCat("Unrecognized action: ", move_id,
                          " for player: ", player);
    }
  }
}

std::string YachtState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

YachtState::YachtState(std::shared_ptr<const Game> game)
    : State(game),
      cur_player_(kChancePlayerId),
      prev_player_(kChancePlayerId),
      turns_(kInitialTurn),
      player1_turns_(0),
      player2_turns_(0),
      dice_({}),
      scores_({0, 0}),
      scoring_sheets_({ScoringSheet(), ScoringSheet()}) {}

Player YachtState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : Player{cur_player_};
}

int YachtState::Opponent(int player) const {
  if (player == kPlayerId1) return kPlayerId2;
  if (player == kPlayerId2) return kPlayerId1;
  SpielFatalError("Invalid player.");
}

void YachtState::RollDie(int outcome) {
  dice_.push_back(kChanceOutcomeValues[outcome - 1]);
}

int YachtState::DiceValue(int i) const {
  SPIEL_CHECK_GE(i, 0);
  SPIEL_CHECK_LT(i, dice_.size());

  if (dice_[i] >= 1 && dice_[i] <= 6) {
    return dice_[i];
  } else if (dice_[i] >= 7 && dice_[i] <= 12) {
    // This die is marked as chosen, so return its proper value.
    // Note: dice are only marked as chosen during the legal moves enumeration.
    return dice_[i] - 6;
  } else {
    SpielFatalError(absl::StrCat("Bad dice value: ", dice_[i]));
  }
}

void YachtState::ApplyNormalAction(Action move, int player) {
  if (move == kFillOnes) {
    scoring_sheets_[player].ones = filled;

    int score = 0;
    for (int i = 0; i < dice_.size(); ++i) {
      int die = dice_[i];
      if (die == 1) {
        score += die;
      }
    }

    scores_[player] += score;
  }
  
  if (move == kFillTwos) {
    scoring_sheets_[player].twos = filled;

    int score = 0;
    for (int i = 0; i < dice_.size(); ++i) {
      int die = dice_[i];
      if (die == 2) {
        score += die;
      }
    }
  }

  scores_[player] += score;

  if (move == kFillThrees) {
    scoring_sheets_[player].threes = filled;

    int score = 0;
      for (int i = 0; i < dice_.size(); ++i) {
        int die = dice_[i];
        if (die == 3) {
          score += die;
        }
      }
  }

  if (move == kFillFours) {
    scoring_sheets_[player].fours = filled;

    int score = 0;
    for (int i = 0; i < dice_.size(); ++i) {
      int die = dice_[i];
      if (die == 4) {
        score += die;
      }
    }
  }

  if (move == kFillFives) {
    scoring_sheets_[player].fives = filled;

    int score = 0;
    for (int i = 0; i < dice_.size(); ++i) {
      int die = dice_[i];
      if (die == 5) {
        score += die;
      }
    }
  }

  if (move == kFillSixes) {
    scoring_sheets_[player].sixes = filled;

    int score = 0;
    for (int i = 0; i < dice_.size(); ++i) {
      int die = dice_[i];
      if (die == 6) {
        score += die;
      }
    }
  }

  if (move == kFillThreeOfAKind) {
    scoring_sheets_[player].three_of_a_kind = filled;

    int score = 0;
    vector<int> vals(6,0);
    bool flag = false;
    for (int i = 0; i < dice_.size(); ++i) {
      int die = dice_[i];
      vals[die]++;
      if (vals[die] >= 3) flag = true;
      score += die;
    }
    if (!flag) score = 0;
  }

  if (move == kFillFourOfAKind) {
    scoring_sheets_[player].four_of_a_kind = filled;

    int score = 0;
    vector<int> vals(6,0);
    bool flag = false;
    for (int i = 0; i < dice_.size(); ++i) {
      int die = dice_[i];
      vals[die]++;
      if (vals[die] >= 4) flag = true;
      score += die;
    }
    if (!flag) score = 0;
  }

  if (move == kFillFullHouse) {
    scoring_sheets_[player].full_house = filled;

    int score = 0;
    vector<int> vals(6,0);
    bool flag3 = false;
    bool flag2 = false;
    for (int i = 0; i < dice_.size(); ++i) {
      int die = dice_[i];
      vals[die]++;
      score += die;
    }
    for (int i = 0; i < 6; ++i) {
      if (vals[i] == 3) flag3=true;
      if (vals[i] == 2) flag2=true;
    }
    if (!flag3 || !flag2) score = 0;
  }

  if (move == kFillLittleStraight) {
    scoring_sheets_[player].little_straight = filled;

    int score = 0;
    vector<int> vals(6,0);
    for (int i = 0; i < dice_.size(); ++i) {
      int die = dice_[i];
      vals[die]++;
    }
    if (vals[0] && vals[1] && vals[2] && vals[3]) score = 30;
    else if (vals[1] && vals[2] && vals[3] && vals[4]) score = 30;
    else if (vals[2] && vals[3] && vals[4] && vals[5]) score = 30;
  }

  if (move == kFillBigStraight) {
    scoring_sheets_[player].big_straight = filled;

    int score = 0;
    vector<int> vals(6,0);
    for (int i = 0; i < dice_.size(); ++i) {
      int die = dice_[i];
      vals[die]++;
    }
    if (vals[0] && vals[1] && vals[2] && vals[3] && vals[4]) score = 40;
    else if (vals[1] && vals[2] && vals[3] && vals[4] && vals[5]) score = 40;
  }

  if (move == kFillYacht) {
    scoring_sheets_[player].yacht = filled;

    int score = 0;
    vector<int> vals(6,0);
    bool flag = false;
    for (int i = 0; i < dice_.size(); ++i) {
      int die = dice_[i];
      vals[die]++;
      if (vals[die] == 5) flag = true;
      score += die;
    }
    if (!flag) score = 0;
  }

  if (move == kFillChoice) {
    scoring_sheets_[player].choice = filled;

    int score = 0;
    for (int i = 0; i < dice_.size(); ++i) {
      int die = dice_[i];
      score += die;
    }
  }
   
} 

void YachtState::IncrementTurn() {
  turns_++;
  if (cur_player_ == kPlayerId1) {
    player1_turns_++;
  } else if (cur_player_ == kPlayerId2) {
    player2_turns_++;
  }

  prev_player_ = cur_player_;
  cur_player_ = kChancePlayerId;

  dice_.clear();
}

void YachtState::DoApplyAction(Action move) {
  if (IsChanceNode()) {
    if (turns_ == kInitialTurn) {
      // First turn.
      SPIEL_CHECK_TRUE(dice_.empty());
      int starting_player = std::rand() % kNumPlayers;
      if (starting_player == 0) {
        // Player1 starts.
        cur_player_ = kChancePlayerId;
        prev_player_ = kPlayerId2;
      } else if (starting_player == 1) {
        // Player2 Starts
        cur_player_ = kChancePlayerId;
        prev_player_ = kPlayerId1;
      } else {
        SpielFatalError(
            absl::StrCat("Invalid starting player: ", starting_player));
      }
      RollDie(move);
      turns_ = 0;
      return;
    } else {
      // Normal chance node.
      SPIEL_CHECK_TRUE(dice_.size() < 5);
      RollDie(move);

      // Once die are done rolling. Set player to non-chance node.
      if (dice_.size() == 5) {
        cur_player_ = Opponent(prev_player_);
      }
      return;
    }
  }

  // Normal action.
  SPIEL_CHECK_TRUE(dice_.size() == 5);

  int player_index = cur_player_ - 1;
  ApplyNormalAction(move, player_index);

  IncrementTurn();
}

bool YachtState::IsPosInHome(int player, int pos) const { return true; }

bool YachtState::UsableDiceOutcome(int outcome) const {
  return (outcome >= 1 && outcome <= 6);
}

std::string YachtState::DiceToString(int outcome) const {
  return std::to_string(outcome);
}

std::vector<Action> YachtState::LegalActions() const {
  if (IsChanceNode()) return LegalChanceOutcomes();
  if (IsTerminal()) return {};

  // TODO(aaronrice): update legal moves for scoring categories and scratches.
  std::vector<Action> legal_actions = {};

  for (int i = 0; i < dice_to_reroll_.size(); i++) {
    bool will_reroll = dice_to_reroll_[i];

    // A player cannot choose a die that has already been chosen to be
    // re-rolled.
    if (!will_reroll) {
      legal_actions.push_back(i + 1);
    }
  }

  // Can choose to be done picking die to re-roll at anytime.
  legal_actions.push_back(kPass);

  return legal_actions;
}

std::vector<std::pair<Action, double>> YachtState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  return kChanceOutcomes;
}

std::string YachtState::ScoringSheetToString(
    const ScoringSheet& scoring_sheet) const {
  std::string result = "";
  absl::StrAppend(&result, "Ones: ");
  absl::StrAppend(&result, scoring_sheet.ones);
  absl::StrAppend(&result, "\n");
  absl::StrAppend(&result, "Twos: ");
  absl::StrAppend(&result, scoring_sheet.twos);
  absl::StrAppend(&result, "\n");
  absl::StrAppend(&result, "Threes: ");
  absl::StrAppend(&result, scoring_sheet.threes);
  absl::StrAppend(&result, "\n");
  absl::StrAppend(&result, "Fours: ");
  absl::StrAppend(&result, scoring_sheet.fours);
  absl::StrAppend(&result, "\n");
  absl::StrAppend(&result, "Five: ");
  absl::StrAppend(&result, scoring_sheet.fives);
  absl::StrAppend(&result, "\n");
  absl::StrAppend(&result, "Sixes: ");
  absl::StrAppend(&result, scoring_sheet.sixes);
  absl::StrAppend(&result, "\n");
  absl::StrAppend(&result, "Three of a Kind: ");
  absl::StrAppend(&result, scoring_sheet.three_of_a_kind);
  absl::StrAppend(&result, "\n");
  absl::StrAppend(&result, "Four of a Kind: ");
  absl::StrAppend(&result, scoring_sheet.four_of_a_kind);
  absl::StrAppend(&result, "\n");
  absl::StrAppend(&result, "Full House: ");
  absl::StrAppend(&result, scoring_sheet.full_house);
  absl::StrAppend(&result, "\n");
  absl::StrAppend(&result, "Little Straight: ");
  absl::StrAppend(&result, scoring_sheet.little_straight);
  absl::StrAppend(&result, "\n");
  absl::StrAppend(&result, "Big Straight: ");
  absl::StrAppend(&result, scoring_sheet.big_straight);
  absl::StrAppend(&result, "\n");
  absl::StrAppend(&result, "Yacht: ");
  absl::StrAppend(&result, scoring_sheet.yacht);
  absl::StrAppend(&result, "\n\n");
  absl::StrAppend(&result, "Choice: ");
  absl::StrAppend(&result, scoring_sheet.choice);
  absl::StrAppend(&result, "\n");
  return result;
}

std::string YachtState::ToString() const {
  std::string state = "";

  absl::StrAppend(&state, "Player 1:\n\n");
  absl::StrAppend(&state, ScoringSheetToString(scoring_sheets_[0]));

  absl::StrAppend(&state, "Player 2:\n\n");
  absl::StrAppend(&state, ScoringSheetToString(scoring_sheets_[1]));

  return state;
}

bool YachtState::IsTerminal() const {
  // A game is over when all players have have filled their scoring sheets.
  const ScoringSheet& player1_scoring_sheet = scoring_sheets_[0];
  if (player1_scoring_sheet.ones == empty ||
      player1_scoring_sheet.twos == empty ||
      player1_scoring_sheet.threes == empty ||
      player1_scoring_sheet.fours == empty ||
      player1_scoring_sheet.fives == empty ||
      player1_scoring_sheet.sixes == empty ||
      player1_scoring_sheet.three_of_a_kind = empty ||
      player1_scoring_sheet.full_house == empty ||
      player1_scoring_sheet.four_of_a_kind == empty ||
      player1_scoring_sheet.little_straight == empty ||
      player1_scoring_sheet.big_straight == empty ||
      player1_scoring_sheet.yacht == empty ||
      player1_scoring_sheet.choice == empty) {
    return false;
  }

  const ScoringSheet& player2_scoring_sheet = scoring_sheets_[1];
  if (player2_scoring_sheet.ones == empty ||
      player2_scoring_sheet.twos == empty ||
      player2_scoring_sheet.threes == empty ||
      player2_scoring_sheet.fours == empty ||
      player2_scoring_sheet.fives == empty ||
      player2_scoring_sheet.sixes == empty ||
      player2_scoring_sheet.three_of_a_kind = empty ||
      player2_scoring_sheet.four_of_a_kind == empty ||
      player2_scoring_sheet.full_house == empty ||
      player2_scoring_sheet.little_straight == empty ||
      player2_scoring_sheet.big_straight == empty ||
      player2_scoring_sheet.yacht == empty ||
      player2_scoring_sheet.choice == empty) {
    return false;
  }

  return true;
}

std::vector<double> YachtState::Returns() const { return {1, 0}; }

std::unique_ptr<State> YachtState::Clone() const {
  return std::unique_ptr<State>(new YachtState(*this));
}

void YachtState::SetState(int cur_player, const std::vector<int>& dice,
                          const std::vector<bool>& dice_to_reroll,
                          const std::vector<int>& scores,
                          const std::vector<ScoringSheet>& scoring_sheets) {
  cur_player_ = cur_player;
  dice_ = dice;
  dice_to_reroll_ = dice_to_reroll;
  scores_ = scores;
  scoring_sheets_ = scoring_sheets;
}

YachtGame::YachtGame(const GameParameters& params) : Game(kGameType, params) {}

}  // namespace yacht
}  // namespace open_spiel
