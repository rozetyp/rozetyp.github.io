import streamlit as st
import requests
import time
import datetime

# Replace with your actual Etherscan API key - Consider using Streamlit secrets for better security
API_KEY = '5WR6W8CWYYM7655C4QMGFH5K8ASPXY1V1X' # Replace with your API Key here

# URLs for different endpoints
GAS_ORACLE_URL = f"https://api.etherscan.io/api?module=gastracker&action=gasoracle&apikey={API_KEY}"
ETH_PRICE_URL = f"https://api.etherscan.io/api?module=stats&action=ethprice&apikey={API_KEY}"
BLOCK_NUMBER_URL = f"https://api.etherscan.io/api?module=proxy&action=eth_blockNumber&apikey={API_KEY}"

def fetch_gas_data():
    """
    Fetches gas fee data from the Etherscan Gas Oracle API.
    Returns a dictionary containing gas prices.
    """
    try:
        response = requests.get(GAS_ORACLE_URL)
        data = response.json()
        if data.get("status") == "1":
            result = data["result"]
            return {
                "SafeGasPrice": result.get("SafeGasPrice"),
                "ProposeGasPrice": result.get("ProposeGasPrice"),
                "FastGasPrice": result.get("FastGasPrice"),
                "SuggestBaseFee": result.get("suggestBaseFee"),
                "GasUsedRatio": result.get("gasUsedRatio")
            }
        else:
            st.error(f"Error fetching gas data: {data.get('message')}")
            return None
    except Exception as e:
        st.error(f"Exception occurred while fetching gas data: {e}")
        return None

def fetch_eth_price_data():
    """
    Retrieves the current ETH price in USD using the Etherscan API.
    """
    try:
        response = requests.get(ETH_PRICE_URL)
        data = response.json()
        if data.get("status") == "1":
            result = data.get("result", {})
            return result.get("ethusd")
        else:
            st.error(f"Error fetching ETH price: {data.get('message')}")
            return None
    except Exception as e:
        st.error(f"Exception occurred while fetching ETH price: {e}")
        return None

def fetch_block_data():
    """
    Retrieves the current block number and basic block details.
    Returns a tuple: (block_number, block_timestamp, gas_used).
    """
    try:
        # First, get the latest block number
        response = requests.get(BLOCK_NUMBER_URL)
        data = response.json()
        if "result" in data:
            block_number_hex = data["result"]
            block_number = int(block_number_hex, 16)
            # Fetch block details using the block number
            block_detail_url = f"https://api.etherscan.io/api?module=proxy&action=eth_getBlockByNumber&tag={block_number_hex}&boolean=true&apikey={API_KEY}"
            block_response = requests.get(block_detail_url)
            block_data = block_response.json()
            if block_data.get("result"):
                result = block_data.get("result")
                block_timestamp_hex = result.get("timestamp")
                block_timestamp = (datetime.datetime.fromtimestamp(int(block_timestamp_hex, 16))
                                                if block_timestamp_hex else None)
                gas_used = int(result.get("gasUsed", "0"), 16) if result.get("gasUsed") else None
                return block_number, block_timestamp, gas_used
            else:
                st.error(f"Error fetching block details: {block_data.get('message')}")
                return None, None, None
        else:
            st.error(f"Error fetching block number: {data.get('message')}")
            return None, None, None
    except Exception as e:
        st.error(f"Exception occurred while fetching block data: {e}")
        return None, None, None

def main():
    st.title("Real-Time Ethereum Data")

    eth_price_placeholder = st.empty()
    gas_prices_placeholder = st.empty()
    block_data_placeholder = st.empty()

    while True:
        eth_price = fetch_eth_price_data()
        gas_data = fetch_gas_data()
        block_number, block_timestamp, gas_used = fetch_block_data()

        if eth_price:
            eth_price_placeholder.metric("ETH Price (USD)", f"${eth_price}")
        if gas_data:
            gas_prices_placeholder.markdown(f"""
                **Gas Prices (Gwei):**
                - Safe: {gas_data.get('SafeGasPrice', 'N/A')}
                - Propose: {gas_data.get('ProposeGasPrice', 'N/A')}
                - Fast: {gas_data.get('FastGasPrice', 'N/A')}
                - Suggest Base Fee: {gas_data.get('SuggestBaseFee', 'N/A')}
                - Gas Used Ratio: {gas_data.get('GasUsedRatio', 'N/A')}
            """)
        if block_number:
            block_data_placeholder.markdown(f"""
                **Latest Block Data:**
                - Block Number: {block_number}
                - Block Timestamp: {block_timestamp.isoformat() if block_timestamp else 'N/A'}
                - Gas Used in Block: {gas_used if gas_used is not None else 'N/A'}
            """)

        time.sleep(30) # Update every 30 seconds - adjust as needed

if __name__ == "__main__":
    main()
