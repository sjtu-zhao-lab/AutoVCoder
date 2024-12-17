import re
import os
import json
import random
import threading

from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

operators = [
    '8-bit serializer', '4-bit adder', '8-bit adder', '16-bit adder', '32-bit adder', '64-bit adder',
    'Carry-lookahead adder', 'Ripple-carry adder', 'Carry-save adder', 'Carry-skip adder',
    'Kogge-Stone adder', 'Brent-Kung adder', 'Wallace tree adder',

    '2-bit multiplier', '4-bit multiplier', '8-bit multiplier', '16-bit multiplier', '32-bit multiplier', '64-bit multiplier',
    'Wallace tree multiplier', 'Booth multiplier', 'Array multiplier',

    'Restoring divider', 'Non-restoring divider', 'SRT divider', '2-bit divider', '4-bit divider', '8-bit divider',

    '2-bit comparator', '4-bit comparator', '8-bit comparator', '16-bit comparator', '32-bit comparator', '64-bit comparator',
    'Magnitude comparator',

    '2-bit counter', '4-bit counter', '8-bit counter', '16-bit counter', '32-bit counter', '64-bit counter',
    'Up counter', 'Down counter', 'Up/Down counter', 'Johnson counter', 'Ring counter',

    '2-to-1 multiplexer', '4-to-1 multiplexer', '8-to-1 multiplexer', '16-to-1 multiplexer', '32-to-1 multiplexer',

    '2-to-4 decoder', '3-to-8 decoder', '4-to-16 decoder', '5-to-32 decoder',
    'BCD-to-7-segment decoder',

    '4-to-2 encoder', '8-to-3 encoder', '16-to-4 encoder', '32-to-5 encoder',
    'Priority encoder',

    'D-latch', 'SR-latch', 'JK-latch', 'T-latch', 'Master-slave D-latch',
    'D-flip-flop', 'SR-flip-flop', 'JK-flip-flop', 'T-flip-flop', 'Edge-triggered D-flip-flop',

    '2-bit shift register', '4-bit shift register', '8-bit shift register', '16-bit shift register',
    'Serial-in serial-out (SISO) shift register', 'Serial-in parallel-out (SIPO) shift register',
    'Parallel-in serial-out (PISO) shift register', 'Parallel-in parallel-out (PIPO) shift register',

    '8-bit CRC generator', '16-bit CRC generator', '32-bit CRC generator',

    '8-bit Hamming encoder', '16-bit Hamming encoder', 'Parity checker', 'Even parity generator', 'Odd parity generator',

    'AND gate', 'OR gate', 'XOR gate', 'NAND gate', 'NOR gate', 'XNOR gate',

    '2-bit barrel shifter', '4-bit barrel shifter', '8-bit barrel shifter',

    '4-tap FIR filter', '8-tap FIR filter', '16-tap FIR filter', '32-tap FIR filter',
    'Butterworth filter', 'Chebyshev filter', '2nd-order IIR filter', '4th-order IIR filter',

    '8-bit ALU', '16-bit ALU', '32-bit ALU', '64-bit ALU',

    '2x PLL', '4x PLL', '8x PLL', 'Fractional-N PLL',

    '1-bit SRAM cell', '4-bit SRAM cell', '8-bit SRAM cell', '16-bit SRAM cell',
    '1-bit DRAM cell', '4-bit DRAM cell', '8-bit DRAM cell', '16-bit DRAM cell',
    'EEPROM cell', 'Flash memory cell',

    '16-bit serializer', '32-bit serializer',
    '8-bit deserializer', '16-bit deserializer', '32-bit deserializer',

    'Round-robin arbiter', 'Priority arbiter', 'Fixed-priority arbiter',

    '2-channel DMA controller', '4-channel DMA controller', '8-channel DMA controller',

    'Divide-by-2 circuit', 'Divide-by-4 circuit', 'Divide-by-8 circuit',

    'Edge detector', 'Pulse width modulator', 'Phase detector', 'Zero-crossing detector',

    'RC oscillator', 'Crystal oscillator', 'Voltage-controlled oscillator (VCO)',

    '8-bit ADC', '16-bit ADC', '24-bit ADC', '8-bit DAC', '16-bit DAC', '24-bit DAC',

    'Linear voltage regulator', 'Switching regulator', 'Low dropout regulator (LDO)',

    'Temperature sensor', 'Voltage sensor', 'Current sensor',

    '8-bit timer', '16-bit timer', '32-bit timer', 'Watchdog timer',

    '2-stage pipeline', '5-stage pipeline', '7-stage pipeline',

    '8-bit AES module', '128-bit AES module', '256-bit AES module', 'RSA module', 'DES module',

    '8-bit LFSR', '16-bit LFSR', '32-bit LFSR', 'Cryptographic RNG',

    'Interrupt controller', 'Reset controller',

    '8-bit data bus', '16-bit data bus', '32-bit data bus',

    '4-point FFT', '8-point FFT', '16-point FFT', '32-point FFT', '64-point FFT',
    'GPIO controller', 'I2C controller', 'SPI controller', 'UART controller', 'PCIe interface',
    'Ethernet MAC controller', 'WiFi module', 'Bluetooth module',
    'SD card interface', 'NVMe controller', 'SATA controller',
    
    "wire", "wire4", "notgate", "andgate", "norgate", "xnorgate",
    "vector0", "vector1", "vector2", "vector3", "vector4", "vector5", "gates4", "vectorgates", "vectorr",
    "alwaysblock1", "alwaysblock2", "always_if", "always_if2", "always_case",
    "conditional", "reduction", "gates100", "vector100r", "popcount255", "adder100i", "bcdadd100",
    "mux2to1", "mux2to1v", "mux9to1v", "mux256to1", "mux256to1v",
    "hadd", "fadd", "adder3", "adder100", "bcdadd4",
    "kmap1", "kmap2", "kmap3", "kmap4",
    "edgedetect", "edgedetect2", "edgecapture",
    "count15", "count10", "count1to10","fsm1", "fsm1s", "fsm2", "fsm2s", "fsm3comb",
    "fsm3onehot", "fsm3", "fsm3s", "ece241_2013_q4", 
    "lemmings1", "lemmings2", "lemmings3", "lemmings4",
    "fsm_onehot", "fsm_ps2", "fsm_ps2data", "fsm_serial", "fsm_serialdata",
    "fsm_serialdp", "fsm_hdlc", "bugs_mux2", "bugs_nand3", "bugs_mux4"
]

algorithms = [
    'Matrix multiplication', 'Vector addition', 'Dot product',
    'Matrix inversion', 'LU decomposition', 'QR decomposition',
    'Cholesky decomposition', 'Polynomial evaluation', 'CORDIC algorithm',
    'Fast Fourier Transform (FFT)', 'Inverse FFT (IFFT)', 
    'Discrete Cosine Transform (DCT)', 'Discrete Wavelet Transform (DWT)',
    'Goertzel algorithm', 'Finite Impulse Response (FIR) filter',
    'Infinite Impulse Response (IIR) filter', 'Adaptive LMS filter',
    'Kalman filter', 'Butterworth filter', 'Chebyshev filter',
    'Sobel edge detection', 'Canny edge detection', 'Gaussian blur',
    'Median filter', 'Histogram equalization', 'Image convolution',
    'Hough transform', 'Morphological operations (Erosion/Dilation)',
    'Thresholding (Otsu’s method)', 'Bilinear interpolation',
    'Nearest neighbor interpolation', 'Color space conversion (RGB to YUV)',
    'Echo cancellation', 'Noise suppression', 'Pitch detection',
    'Audio mixing', 'Sample rate conversion', 'Dynamic range compression',
    'Viterbi decoder', 'Turbo encoder/decoder', 'Reed-Solomon encoder',
    'Convolutional encoder', 'Hamming encoder', 'Cyclic Redundancy Check (CRC)',
    'Quadrature Amplitude Modulation (QAM)', 'Orthogonal Frequency Division Multiplexing (OFDM)',
    'Phase-Locked Loop (PLL)', 'Carrier frequency recovery',
    'AES encryption', 'DES encryption', 'RSA encryption', 
    'SHA-256 hash function', 'MD5 hash function', 'Elliptic Curve Cryptography (ECC)',
    'ChaCha20 stream cipher', 'HMAC (Hash-based Message Authentication Code)',
    'Linear regression', 'Logistic regression', 'K-means clustering',
    'K-Nearest Neighbors (KNN)', 'Support Vector Machine (SVM)',
    'Decision tree inference', 'Convolutional Neural Network (CNN) inference',
    'Recurrent Neural Network (RNN) inference', 'Matrix factorization for recommendation',
    'Gradient descent optimization', 'Backpropagation for neural networks',
    'Binary search', 'Merge sort', 'Quick sort', 
    'Heap sort', 'Bubble sort', 'Insertion sort',
    'Proportional-Integral-Derivative (PID) control',
    'State-space control', 'Model predictive control (MPC)',
    'Finite State Machine (FSM)', 'Deadbeat control',
    'Huffman encoding', 'Lempel-Ziv-Welch (LZW) compression',
    'Run-length encoding (RLE)', 'JPEG compression',
    'PNG compression', 'Audio compression (MP3/AAC)',
    'IP checksum calculation', 'TCP segmentation',
    'Packet filtering (Firewall)', 'Network Address Translation (NAT)',
    'Black-Scholes model', 'Monte Carlo simulation',
    'Option pricing', 'Portfolio optimization',
    'A* search algorithm', 'Dijkstra’s algorithm', 'RRT (Rapidly-exploring Random Tree)',
    'SLAM (Simultaneous Localization and Mapping)', 'Kalman filter for sensor fusion',
    'FIFO queue', 'LIFO stack', 'Binary heap', 'Hash table',
    'Linked list traversal', 'Binary search tree operations',
    'Fixed-point multiplication', 'Floating-point multiplication',
    'Fixed-point addition', 'Floating-point addition',
    'Linear Feedback Shift Register (LFSR)', 'Mersenne Twister',
    'Cryptographically Secure Pseudo-Random Number Generator (CSPRNG)',
    'Breadth-First Search (BFS)', 'Depth-First Search (DFS)',
    'Dijkstra’s shortest path', 'Bellman-Ford algorithm',
    'Floyd-Warshall algorithm', 'Prim’s Minimum Spanning Tree (MST)',
    'Kruskal’s MST algorithm', 'Graph coloring algorithm',
    'Flow control (Stop-and-wait)', 'Sliding window protocol',
    'Error correction with ARQ', 'Data integrity with checksum',
    'Clock divider', 'Digital phase shifter', 'Watchdog timer',
    'Pulse generator', 'Frequency divider',
    'I2C master/slave controller', 'SPI master/slave controller',
    'UART transmitter/receiver', 'Ethernet MAC controller',
    'PCIe endpoint controller',
    'Instruction pipeline', 'Data forwarding',
    'Hazard detection and resolution', 'Branch prediction',
    'Dynamic Voltage and Frequency Scaling (DVFS)',
    'Clock gating', 'Power gating',
    'Thermal sensor calibration', 'Fan speed control',
    'Temperature-based shutdown',
    'Memory allocation (malloc/free)', 'Cache replacement policies (LRU, FIFO)',
    'Virtual memory management', 'Direct Memory Access (DMA)',
]

problems = algorithms + operators

def query_gpt4omini(client, prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    llm_response = client.chat.completions.create(
        messages=messages,
        model="gpt-4o-mini",
        temperature=0.8,
        stream=False
    )
    llm_outputs = llm_response.choices[0].message.content  
    
    circuit_type_match = re.search(r'<ct>(.*?)</ct>', llm_outputs, re.DOTALL)
    if circuit_type_match:
        circuit_type = circuit_type_match.group(1)
    else:
        raise ValueError("No circuit type found in the response.")
    
    problem_match = re.search(r'<vp>(.*?)</vp>', llm_outputs, re.DOTALL)
    if problem_match:
        problem = problem_match.group(1)
    else:
        raise ValueError("No Verilog problem found in the response.")
    
    verilog_code_match = re.search(r'<v>(.*?)</v>', llm_outputs, re.DOTALL)
    if verilog_code_match:
        verilog_code = re.search(r'module.*?endmodule', verilog_code_match.group(1), re.DOTALL).group(0)
    else:
        raise ValueError("No Verilog code found in the response.")
    
    verilog_testbench_match = re.search(r'<tb>(.*?)</tb>', llm_outputs, re.DOTALL)
    if verilog_testbench_match:
        verilog_testbench = re.search(r'module.*?endmodule', verilog_testbench_match.group(1), re.DOTALL).group(0)
    else:
        raise ValueError("No Verilog testbench found in the response.")
    
    python_code_match = re.search(r'<py>(.*?)</py>', llm_outputs, re.DOTALL)
    if python_code_match:
        python_code_content = python_code_match.group(1)
        python_code_twice_match = re.search(r'```python(.*?)```', python_code_content, re.DOTALL)
        if python_code_twice_match:
            python_code = python_code_twice_match.group(1).strip()
        else:
            # 如果没有找到 ```python``` 块，可能需要处理或记录这个情况
            python_code = python_code_content.strip()
    else:
        raise ValueError("No Python code found in the response.")
    
    return circuit_type, problem, verilog_code, verilog_testbench, python_code

def write_to_file(data, lock):
    with lock:
        with open('../../../data/second_round/origin2.jsonl', 'a') as f:
            f.write(json.dumps(data) + '\n')

def process_code(client, id, lock):
    level = random.choice(['easy', 'medium', 'hard'])
    problem_type = random.choice(problems)
    
    prompt = f'''This is No.{id} problem. You make sure your Verilog code can be simply simulated and implemented.
You are going to create a {level} Verilog problem for me. This problem related to {problem_type} type. Please answer this question in five parts:
1. Problem circuit type, combinational logic or sequential logic, start with <ct> end with </ct>,
2. Verilog problem, start with <vp> end with </vp>,
3. Verilog code to solve this problem, start with <v> end with </v>
4. Verilog testbench, if sequential logic, write testcase by yourself and don't read from testcases; if combinational logic, you must! and must! and must! read input and standard output from {id}_testcase.txt and verify it. You must both dump 'Test is OK!’ if testbench is pass. It will contain 5 testcases, start with <tb> end with </tb>,
5. Python code equivalent to the Verilog code, main function should cantain 5 testcase and print input and standard output from the python api to {id}_testcase.txt, the test file output by Python and the input read by Verilog should be consistent and easy to parse in format, don't write any prompt to the testcase file like 'Input:' or 'Output:', an exmaple '00000 11111 2222\n00000 1111 2222\n' is really good, don't assert in Python code, start with <py> end with </py>,
Note that your answer should only contain these five parts and no other response.'''
    
    try:
        circuit_type, problem, verilog_code, verilog_testbench, python_code = query_gpt4omini(client, prompt)
        data = {
            "circuit_type": circuit_type.lower(),
            "problem": problem,
            "verilog_code": verilog_code,
            "verilog_testbench": verilog_testbench,
            "python_code": python_code,
            "id": id,
            "level": level,
            "problem_type": problem_type
        }
        write_to_file(data, lock)
    except Exception as e:
        print(f"Error processing code {id}: {e}")
    
def main():
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if openai_api_key:
        print("OpenAI API Key successfully retrieved.")
    else:
        print("Failed to retrieve OpenAI API Key.")
    openai_api_base = "https://a.fe8.cn/v1"
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    
    with open('../../../data/second_round/origin2.jsonl', 'w') as f:
        f.write('')
    
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_code, client, i, lock) for i in range(120001, 130000)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            future.result()

if __name__ == "__main__":
    main()