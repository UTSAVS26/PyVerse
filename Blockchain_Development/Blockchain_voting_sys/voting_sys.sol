// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Voting {
    struct Candidate {
        uint id;
        string name;
        uint voteCount;
    }

    address public owner;
    mapping(uint => Candidate) public candidates;
    mapping(address => bool) public voters;
    uint public candidatesCount;
    bool public votingOpen;

    event VotedEvent(uint indexed candidateId);
    event CandidateAdded(uint indexed candidateId, string name);
    event VotingStatusChanged(bool status);

    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can perform this action.");
        _;
    }

    modifier whenVotingOpen() {
        require(votingOpen, "Voting is not open.");
        _;
    }

    constructor() {
        owner = msg.sender;
        addCandidate("Alice");
        addCandidate("Bob");
        votingOpen = false;
    }

    function addCandidate(string memory _name) public onlyOwner {
        candidatesCount++;
        candidates[candidatesCount] = Candidate(candidatesCount, _name, 0);
        emit CandidateAdded(candidatesCount, _name);
    }

    function openVoting() public onlyOwner {
        votingOpen = true;
        emit VotingStatusChanged(votingOpen);
    }

    function closeVoting() public onlyOwner {
        votingOpen = false;
        emit VotingStatusChanged(votingOpen);
    }

    function vote(uint _candidateId) public whenVotingOpen {
        require(!voters[msg.sender], "You have already voted.");
        require(_candidateId > 0 && _candidateId <= candidatesCount, "Invalid candidate ID.");

        voters[msg.sender] = true;
        candidates[_candidateId].voteCount++;
        emit VotedEvent(_candidateId);
    }
}
