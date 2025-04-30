// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@aave/core-v3/contracts/flashloan/base/FlashLoanReceiverBase.sol";
import "@aave/core-v3/contracts/interfaces/IPoolAddressesProvider.sol";

import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

interface IGasPriceOracle {
    function latestAnswer() external view returns (int256);
    function decimals() external view returns (uint8);
}

interface IUniswapV2Router {
    function swapExactTokensForTokens(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external returns (uint[] memory amounts);

    function getAmountsOut(uint amountIn, address[] calldata path)
        external
        view
        returns (uint[] memory amounts);
}

contract SimpleFlashLoan is FlashLoanReceiverBase {
    using SafeERC20 for IERC20;

    // ------------------------------------------------------------------ //
    //                               STATE                                //
    // ------------------------------------------------------------------ //

    address payable public immutable owner;
    IGasPriceOracle public immutable gasPriceOracle;

    uint256 public maxGasPriceWei;
    address public immutable uniswapRouter;
    address public immutable sushiswapRouter;

    bool private locked;

    enum Strategy {
        SWAP
    }

    // ------------------------------------------------------------------ //
    //                                EVENTS                              //
    // ------------------------------------------------------------------ //

    event FlashLoanRequested(address indexed asset, uint256 amount);
    event SwapExecuted(
        address indexed assetBorrowed,
        uint256 amountBorrowed,
        uint256 premium,
        uint256 amountReturned,
        uint256 profit
    );
    event OperationFailed(string reason);
    event MaxGasPriceUpdated(uint256 newMaxGasPriceWei);
    event ProfitWithdrawn(address indexed to, uint256 amount);

    // ------------------------------------------------------------------ //
    //                               ERRORS                               //
    // ------------------------------------------------------------------ //

    error NotOwner();
    error Reentrancy();
    error GasTooHigh(uint256 currentWei, uint256 capWei);
    error InvalidAsset();
    error AmountZero();
    error UnsupportedStrategy();

    // ------------------------------------------------------------------ //
    //                              MODIFIERS                             //
    // ------------------------------------------------------------------ //

    modifier onlyOwner() {
        if (msg.sender != owner) revert NotOwner();
        _;
    }

    modifier nonReentrant() {
        if (locked) revert Reentrancy();
        locked = true;
        _;
        locked = false;
    }

    // ------------------------------------------------------------------ //
    //                            CONSTRUCTOR                             //
    // ------------------------------------------------------------------ //

    constructor(
        address _provider,
        address _gasPriceOracle,
        address _uniswapRouter,
        address _sushiswapRouter
    ) FlashLoanReceiverBase(IPoolAddressesProvider(_provider)) {
        owner = payable(msg.sender);
        gasPriceOracle = IGasPriceOracle(_gasPriceOracle);
        maxGasPriceWei = type(uint256).max;
        uniswapRouter = _uniswapRouter;
        sushiswapRouter = _sushiswapRouter;
    }

    // ------------------------------------------------------------------ //
    //                       EXTERNAL / PUBLIC API                        //
    // ------------------------------------------------------------------ //

    /// @notice Initiate a flash-loan for a token-swap arbitrage.
    function requestFlashLoan(
        address asset,
        uint256 amount,
        address[] calldata path,
        uint256 minOut,
        bool useUniswap
    ) external onlyOwner nonReentrant {
        if (asset == address(0)) revert InvalidAsset();
        if (amount == 0) revert AmountZero();
        if (path.length < 2 || path[0] != asset)
            revert("Invalid swap path");

        bytes memory data = abi.encode(Strategy.SWAP, path, minOut, useUniswap);

        address;
        assets[0] = asset;
        uint256;
        amounts[0] = amount;
        uint256; // 0 = no debt (full repay)

        emit FlashLoanRequested(asset, amount);

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

    /// @inheritdoc IFlashLoanReceiver
    function executeOperation(
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata premiums,
        address,
        bytes calldata params
    ) external override nonReentrant returns (bool) {
        if (msg.sender != address(LENDING_POOL))
            revert("Caller not Aave pool");

        // Gas guard
        uint256 currentWei = _currentGasPriceWei();
        if (currentWei > maxGasPriceWei)
            revert GasTooHigh(currentWei, maxGasPriceWei);

        address asset = assets[0];
        uint256 amount = amounts[0];
        uint256 premium = premiums[0];

        (Strategy strat, address[] memory path, uint256 minOut, bool useUni) =
            abi.decode(params, (Strategy, address[], uint256, bool));

        if (strat == Strategy.SWAP) {
            IERC20(asset).safeApprove(address(LENDING_POOL), amount + premium);

            try
                this._executeSwap(asset, amount, premium, path, minOut, useUni)
            returns (uint256 returned, uint256 profit) {
                emit SwapExecuted(asset, amount, premium, returned, profit);
            } catch Error(string memory reason) {
                emit OperationFailed(reason);
                return false;
            } catch {
                emit OperationFailed("Swap failed");
                return false;
            }
        } else {
            revert UnsupportedStrategy();
        }

        return true;
    }

    /// @notice Withdraw any ERC-20 balance (profit) to owner.
    function withdrawToken(address token) external onlyOwner {
        uint256 bal = IERC20(token).balanceOf(address(this));
        if (bal == 0) revert("Zero balance");
        IERC20(token).safeTransfer(owner, bal);
        emit ProfitWithdrawn(owner, bal);
    }

    /// @notice Withdraw native ETH (if any) to owner.
    function withdrawETH() external onlyOwner {
        uint256 bal = address(this).balance;
        if (bal == 0) revert("Zero balance");
        (bool ok, ) = owner.call{value: bal}("");
        require(ok, "ETH transfer failed");
        emit ProfitWithdrawn(owner, bal);
    }

    /// @notice Owner can tighten / loosen the gas cap.
    function setMaxGasPriceWei(uint256 newCapWei) external onlyOwner {
        maxGasPriceWei = newCapWei;
        emit MaxGasPriceUpdated(newCapWei);
    }

    // ------------------------------------------------------------------ //
    //                        INTERNAL FUNCTIONS                           //
    // ------------------------------------------------------------------ //

    function _executeSwap(
        address asset,
        uint256 amount,
        uint256 premium,
        address[] memory path,
        uint256 minOut,
        bool useUni
    ) external returns (uint256 returned, uint256 profit) {
        // Only callable internally through DELEGATECALL protection
        if (msg.sender != address(this)) revert("Self-call only");

        address router = useUni ? uniswapRouter : sushiswapRouter;

        // Grant allowance only for this tx
        IERC20(asset).approve(router, amount);

        uint[] memory outs = IUniswapV2Router(router).swapExactTokensForTokens(
            amount,
            minOut,
            path,
            address(this),
            block.timestamp + 300
        );

        // Reset allowance to zero for safety
        IERC20(asset).approve(router, 0);

        returned = outs[outs.length - 1];
        uint256 debt = amount + premium;
        require(returned >= debt, "Not profitable");

        profit = returned - debt;
        // leave profit in contract; owner can withdraw later
    }

    function _currentGasPriceWei() internal view returns (uint256) {
        int256 answer = gasPriceOracle.latestAnswer();
        uint8 dec = gasPriceOracle.decimals();
        require(answer > 0, "Oracle error");
        // Normalise to Wei (1e9 Gwei)
        uint256 scaled = uint256(answer);
        if (dec < 9) {
            scaled = scaled * (10 ** (9 - dec));
        } else if (dec > 9) {
            scaled = scaled / (10 ** (dec - 9));
        }
        return scaled;
    }

    // ------------------------------------------------------------------ //
    //                              FALLBACKS                             //
    // ------------------------------------------------------------------ //

    receive() external payable {}
    fallback() external payable {}
}
