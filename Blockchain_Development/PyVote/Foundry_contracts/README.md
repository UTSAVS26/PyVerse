## Voting.sol
User deploys a contract as owner. User can add candidates for Voting who will be provided with array index as their id. Other users can vote once on a candidate of their choice.

## Foundry

**Foundry is a blazing fast, portable and modular toolkit for Ethereum application development written in Rust.**

Foundry consists of:

-   **Forge**: Ethereum testing framework (like Truffle, Hardhat and DappTools).
-   **Cast**: Swiss army knife for interacting with EVM smart contracts, sending transactions and getting chain data.
-   **Anvil**: Local Ethereum node, akin to Ganache, Hardhat Network.
-   **Chisel**: Fast, utilitarian, and verbose solidity REPL.


## Useful Commands For this Project

### Compiling 

```shell
forge compile
```

### Test

```shell
forge test
```

### Anvil

```shell
anvil
```

### Deploy

```shell
forge script script/Deploy.s.sol --rpc-url <your_rpc_url> --private-key <your_private_key>
```

### Sample execution (Be within `Foundry_contracts`)

1) Compile using :
 ```shell
forge compile
```
2) (Optional) Test using :
```shell
forge test
```
Note: It will automatically compile on test if not done so. 

3) Split terminal into 2.
4) Use following command to spin up a local-chain to work on:
```shell
anvil
```
5) Make sure anvil is working.
6) Deploy using : 
```shell
forge script script/Deploy.s.sol --rpc-url <your_rpc_url> --private-key <your_private_key> --broadcast
```
Note: Sample command provided in script/Deploy.s.sol as comment. You can get private key and rpc url from anvil itself.


## Documentation

https://book.getfoundry.sh/


### Help

```shell
$ forge --help
$ anvil --help
$ cast --help
```


