// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.27;

import {Script, console} from "forge-std/Script.sol";
import {Voting} from "../src/Voting.sol";

contract Deploy is Script {
    Voting public voting;

    function run() public returns(Voting) {
        // Function caller deploys Voting contract
        vm.startBroadcast(msg.sender);
        voting = new Voting();
        
        voting.addCandidate();
        voting.addCandidate();
        voting.addCandidate();
        voting.addCandidate();

        vm.stopBroadcast();
        console.log(voting.owner());
        return (voting);
    }
}


// To deploy on anvil ( local chain ) with anvil's private key:
// forge script script/Deploy.s.sol --fork-url http://localhost:8545 --private-key 0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80 --broadcast

