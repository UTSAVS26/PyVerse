// SPDX-License-Identifier: MIT
pragma solidity ^0.8.27;

contract Voting {

// owner of contract
address public owner;

// Users who have voted or not
mapping(address => bool) public Voted;

// Array to count number of votes for each candidate
// Candidate Id = Array index
uint256[] public candidates;

// Constructor
constructor() {
  owner = msg.sender;
}

// Errors
error Voting__NotOwner();
error Voting__AlreadyVoted();
error Voting__IncorrectVoteIndex();

// Modifier to check if the caller is the owner
modifier ownerOnly(){
 require(msg.sender == owner, Voting__NotOwner());
 _;
}

// Owner is able to add candidatest to the election
function addCandidate() public ownerOnly {
  candidates.push(0);
}

// Any user can vote only once to a valid candidate
function vote(uint _voteIndex) public {
  require(!Voted[msg.sender], Voting__AlreadyVoted());
  require(_voteIndex<candidates.length, Voting__IncorrectVoteIndex());
  Voted[msg.sender] = true;
  candidates[_voteIndex] += 1;
}

// Get candidates array which shows the votes gotten by each Candidate
function getVotes() public view returns (uint256[] memory) {
  return candidates;
}

}
