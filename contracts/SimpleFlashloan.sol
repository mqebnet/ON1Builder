pragma solidity ^0.8.20;

import "https://github.com/aave/aave-v3-core/blob/master/contracts/flashloan/base/FlashLoanSimpleReceiverBase.sol";
import "https://github.com/aave/aave-v3-core/blob/master/contracts/interfaces/IPoolAddressesProvider.sol";
import "https://github.com/aave/aave-v3-core/blob/master/contracts/dependencies/openzeppelin/contracts/IERC20.sol";

interface IGasPriceOracle {
    function latestAnswer() external view returns (int256);
}

contract SimpleFlashLoan is FlashLoanSimpleReceiverBase {
    address payable public owner;
    IGasPriceOracle public gasPriceOracle;
    bool private locked;

    event FlashLoanRequested(address token, uint256 amount);
    event FlashLoanExecuted(address token, uint256 amount, uint256 premium, bool success);
    event FlashLoanFailed(address token, uint256 amount, uint256 premium, string reason);

    constructor(address _addressProvider, address _gasPriceOracle) FlashLoanSimpleReceiverBase(IPoolAddressesProvider(_addressProvider)) {
        owner = payable(msg.sender); // Contract deployer becomes the owner
        gasPriceOracle = IGasPriceOracle(_gasPriceOracle);
        locked = false;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Not contract owner");
        _;
    }

    modifier nonReentrant() {
        require(!locked, "Reentrant call");
        locked = true;
        _;
        locked = false;
    }

    function fn_RequestFlashLoan(address[] memory _tokens, uint256[] memory _amounts) public {
        require(_tokens.length == _amounts.length, "Tokens and amounts length mismatch");
        for (uint256 i = 0; i < _tokens.length; i++) {
            address receiverAddress = address(this);
            address asset = _tokens[i];
            uint256 amount = _amounts[i];
            bytes memory params = "";  
            uint16 referralCode = 0;  

            emit FlashLoanRequested(_tokens[i], _amounts[i]);
            
            POOL.flashLoanSimple(
                receiverAddress,
                asset,
                amount,
                params,
                referralCode
            );
        }
    }

    function executeOperation(
        address asset,
        uint256 amount,
        uint256 premium,
        address initiator,
        bytes calldata params
    ) external override nonReentrant returns (bool) {
        try {
            // Add your flash loan logic here

            emit FlashLoanExecuted(asset, amount, premium, true);
            return true;
        } catch (bytes memory reason) {
            emit FlashLoanFailed(asset, amount, premium, string(reason));
            return false;
        }
    }

    function withdrawToken(address _tokenAddress) public onlyOwner {
        IERC20 token = IERC20(_tokenAddress);
        uint256 balance = token.balanceOf(address(this));
        require(balance > 0, "No tokens to withdraw");
        token.transfer(owner, balance);
    }

    function withdrawETH() public onlyOwner {
        uint256 balance = address(this).balance;
        require(balance > 0, "No ETH to withdraw");
        owner.transfer(balance);
    }

    receive() external payable {}
}
