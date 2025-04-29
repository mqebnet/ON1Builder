// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "https://github.com/aave/aave-v3-core/blob/master/contracts/flashloan/base/FlashLoanReceiverBase.sol";
import "https://github.com/aave/aave-v3-core/blob/master/contracts/interfaces/IPoolAddressesProvider.sol";
import "https://github.com/aave/aave-v3-core/blob/master/contracts/dependencies/openzeppelin/contracts/SafeERC20.sol";
import "https://github.com/aave/aave-v3-core/blob/master/contracts/dependencies/openzeppelin/contracts/IERC20.sol";

interface IGasPriceOracle {
    function latestAnswer() external view returns (int256);
}

interface IUniswapV2Router {
    function swapExactTokensForTokens(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external returns (uint[] memory amounts);
    function getAmountsOut(uint amountIn, address[] calldata path) external view returns (uint[] memory amounts);
}

contract SimpleFlashLoan is FlashLoanReceiverBase {
    using SafeERC20 for IERC20;

    address payable public owner;
    IGasPriceOracle public gasPriceOracle;
    uint256 public maxGasPrice;
    address public uniswapRouter;
    address public sushiswapRouter;
    bool private locked;

    enum Strategy { SWAP }

    event FlashLoanRequested(address[] assets, uint256[] amounts);
    event SwapExecuted(address assetBorrowed, uint256 amountBorrowed, uint256 premium, uint256 amountReturned, uint256 profit);
    event OperationFailed(string reason);
    event MaxGasPriceUpdated(uint256 newMaxGasPrice);
    event ProfitWithdrawn(address to, uint256 amount);

    constructor(
        address _provider,
        address _gasPriceOracle,
        address _uniswapRouter,
        address _sushiswapRouter
    ) FlashLoanReceiverBase(IPoolAddressesProvider(_provider)) {
        owner = payable(msg.sender);
        gasPriceOracle = IGasPriceOracle(_gasPriceOracle);
        maxGasPrice = type(uint256).max;
        uniswapRouter = _uniswapRouter;
        sushiswapRouter = _sushiswapRouter;
        locked = false;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier nonReentrant() {
        require(!locked, "Reentrancy");
        locked = true;
        _;
        locked = false;
    }

    /**
     * @notice Request a flashloan for a single asset swap strategy
     * @param asset The token to borrow
     * @param amount The borrow amount
     * @param path Swap path including borrowed asset and target token
     * @param minOut Minimum output amount for swap
     * @param useUniswap If true use Uniswap, else use SushiSwap
     */
    function requestFlashLoan(
        address asset,
        uint256 amount,
        address[] calldata path,
        uint256 minOut,
        bool useUniswap
    ) external onlyOwner nonReentrant {
        require(asset != address(0), "Invalid asset");
        require(amount > 0, "Amount must be > 0");
        bytes memory data = abi.encode(Strategy.SWAP, path, minOut, useUniswap);
        address[] memory assets = new address[](1);
        assets[0] = asset;
        uint256[] memory amounts = new uint256[](1);
        amounts[0] = amount;
        uint256[] memory modes = new uint256[](1);
        modes[0] = 0; // no debt, full repay
        emit FlashLoanRequested(assets, amounts);
        LENDING_POOL.flashLoan(
            address(this),
            assets,
            amounts,
            modes,
            address(this),
            data,
            0
        );
    }

    /**
     * @notice Aave callback after flashloan
     */
    function executeOperation(
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata premiums,
        address,
        bytes calldata params
    ) external override nonReentrant returns (bool) {
        require(msg.sender == address(LENDING_POOL), "Unauthorized caller");
        uint256 currentGas = uint256(gasPriceOracle.latestAnswer());
        require(currentGas <= maxGasPrice, "Gas price too high");

        // Only one asset supported
        address asset = assets[0];
        uint256 amount = amounts[0];
        uint256 premium = premiums[0];

        (Strategy strat, address[] memory path, uint256 minOut, bool useUniswap) = abi.decode(
            params,
            (Strategy, address[], uint256, bool)
        );

        if (strat == Strategy.SWAP) {
            // Approve pool to pull repay
            IERC20(asset).safeApprove(address(LENDING_POOL), amount + premium);
            try this._executeSwap(asset, amount, premium, path, minOut, useUniswap) returns (uint256 returned, uint256 profit) {
                emit SwapExecuted(asset, amount, premium, returned, profit);
            } catch Error(string memory reason) {
                emit OperationFailed(reason);
                return false;
            } catch {
                emit OperationFailed("Swap failed");
                return false;
            }
        } else {
            emit OperationFailed("Unsupported strategy");
            return false;
        }
        return true;
    }

    /**
     * @notice Internal execution of swap strategy
     */
    function _executeSwap(
        address asset,
        uint256 amount,
        uint256 premium,
        address[] memory path,
        uint256 minOut,
        bool useUniswap
    ) external returns (uint256 returned, uint256 profit) {
        require(msg.sender == address(this), "Caller must be self");
        address router = useUniswap ? uniswapRouter : sushiswapRouter;
        IERC20(asset).safeApprove(router, amount);
        uint[] memory amountsOut = IUniswapV2Router(router).swapExactTokensForTokens(
            amount,
            minOut,
            path,
            address(this),
            block.timestamp + 300
        );
        returned = amountsOut[amountsOut.length - 1];
        uint256 debt = amount + premium;
        require(returned >= debt, "Returned < debt");
        profit = returned - debt;
        // Approve any surplus for owner withdraw
        IERC20(path[path.length - 1]).safeApprove(owner, profit);
        return (returned, profit);
    }

    /**
     * @notice Withdraw ERC20 profits
     */
    function withdrawToken(address token) external onlyOwner {
        uint256 bal = IERC20(token).balanceOf(address(this));
        require(bal > 0, "No token balance");
        IERC20(token).safeTransfer(owner, bal);
        emit ProfitWithdrawn(owner, bal);
    }

    /**
     * @notice Withdraw ETH profits
     */
    function withdrawETH() external onlyOwner {
        uint256 bal = address(this).balance;
        require(bal > 0, "No ETH balance");
        owner.transfer(bal);
        emit ProfitWithdrawn(owner, bal);
    }

    /**
     * @notice Set max gas price cap
     */
    function setMaxGasPrice(uint256 _maxGasPrice) external onlyOwner {
        maxGasPrice = _maxGasPrice;
        emit MaxGasPriceUpdated(_maxGasPrice);
    }

    receive() external payable {}
    fallback() external payable {}
}
