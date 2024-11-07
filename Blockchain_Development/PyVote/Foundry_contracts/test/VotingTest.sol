// SPDX-License-Identifier: MIT
pragma solidity ^0.8.27;

import "forge-std/Test.sol";
import "../src/Voting.sol";

contract VotingTest is Test {
    Voting voting;
    address owner = makeAddr("Owner");
    address voter1 = makeAddr("Voter1");

    function setUp() public {
        vm.prank(owner);
        voting = new Voting();
    }

    function testOwnerIsCorrect() public view{
        assertEq(voting.owner(), owner);
    }

    function testAddCandidateAsOwner() public {
        vm.startPrank(owner);
        voting.addCandidate();
        voting.addCandidate();
        vm.stopPrank();

        uint256[] memory candidates = voting.getVotes();
        assertEq(candidates.length, 2);
    }

    function testAddCandidateNotOwner() public {
        vm.expectRevert(Voting.Voting__NotOwner.selector);
        voting.addCandidate();
    }

    function testVote() public {
        vm.prank(owner);
        voting.addCandidate();

        vm.prank(voter1);
        voting.vote(0);

        uint256[] memory candidates = voting.getVotes();
        assertEq(candidates.length, 1);
        assertEq(candidates[0], 1);
    }

    function testVoteTwice() public {
        vm.prank(owner);
        voting.addCandidate();

        vm.startPrank(voter1);
        voting.vote(0);
        vm.expectRevert(Voting.Voting__AlreadyVoted.selector);
        voting.vote(0);
        vm.stopPrank();
    }

    function testVoteWithInvalidCandidate() public {
        vm.prank(voter1);
        vm.expectRevert(Voting.Voting__IncorrectVoteIndex.selector);
        voting.vote(100);
    }
}
