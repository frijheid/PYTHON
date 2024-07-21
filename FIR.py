import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, savgol_filter

def on_click(event, xdata, ydata, ax, fig):
    if event.inaxes:  # Check if the click is inside the plot area
        # Find the closest point
        distances = np.sqrt((xdata - event.xdata) ** 2 + (ydata - event.ydata) ** 2)
        min_index = np.argmin(distances)
        x_val = xdata[min_index]
        y_val = ydata[min_index]

        # Access stored markers and annotations
        markers = fig.__dict__.get('markers', [])
        annotations = fig.__dict__.get('annotations', [])
        
        # Check if this point is already in the list
        for marker, annotation in zip(markers, annotations):
            if np.isclose(marker.get_xdata(), x_val) and np.isclose(marker.get_ydata(), y_val):
                # Remove the existing marker and annotation
                marker.remove()
                annotation.remove()
                markers.remove(marker)
                annotations.remove(annotation)
                fig.__dict__['markers'] = markers
                fig.__dict__['annotations'] = annotations
                plt.draw()
                return
        
        # Add a new marker and annotation
        marker, = ax.plot(x_val, y_val, 'ro')  # Red dot
        annotation = ax.annotate(f'({x_val:.2f}, {y_val:.2f})',
                                 xy=(x_val, y_val),
                                 xytext=(x_val + 0.1, y_val + 0.1),
                                 textcoords='offset points',
                                 arrowprops=None)
        markers.append(marker)
        annotations.append(annotation)
        fig.__dict__['markers'] = markers
        fig.__dict__['annotations'] = annotations
        
        plt.draw()

def plot_signal(fig_title, x, y, x_label, y_label, fig_size=(8, 4)):  # Reduced size
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(x, y, marker='o')
    ax.set_title(fig_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid()

    # Initialize lists for markers and annotations
    fig.__dict__['markers'] = []
    fig.__dict__['annotations'] = []

    # Connect the click event to the callback function
    fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, x, y, ax, fig))

    plt.show(block=False)  # Use non-blocking mode to allow other plots to be displayed

def main():
    # %% PREPARATION
    # Parameters
    N = 65
    t = np.linspace(0, 1, N)
    # Generate a sinusoidal signal with noise
    signal = np.sin(2 * np.pi * 5 * t) + np.random.normal(0, 0.5, N)

    # %% FILTERING MAK NYUS
    # Butterworth filter parameters
    applyFiltering = True      # Set to true to apply filtering, false to skip filtering
    filterType = 'bandpass'    # Choose 'low', 'high', 'bandpass', or 'stop'
    cutoffFrequency1 = 3     # Example cutoff frequency for lowpass and highpass
    cutoffFrequency2 = 6       # Example upper cutoff frequency for bandpass and stop
    filterOrder = 4            # Example filter order
    check_frequency = True     # Set to true to perform FFT analysis before and after processing

    # Smoothing parameters
    applySmoothing = True     # Set to true to apply smoothing
    smoothingMethod = 'exponential'  # Choose 'moving', 'exponential', or 'savitzky-golay'
    movingAverageWindow = 1000000    # Example window size for moving average
    exponentialAlpha = 0.0001  # Smoothing factor for exponential smoothing
    savitzkyOrder = 3          # Polynomial order for Savitzky-Golay filter
    savitzkyFrameLength = 11   # Frame length for Savitzky-Golay filter

    # Calculate the sampling period and sampling frequency
    T = t[1] - t[0]
    Fs = 1 / T

    # Length of the signal
    L = len(signal)

    if check_frequency:
        # Perform the FFT before processing
        Y = np.fft.fft(signal)

        # Compute the two-sided spectrum and then the single-sided spectrum
        P2 = np.abs(Y / L)
        P1 = P2[:L // 2 + 1]
        P1[1:-1] = 2 * P1[1:-1]

        # Define the frequency domain f
        f = Fs * np.arange(L // 2 + 1) / L

        # Plot the single-sided amplitude spectrum before processing
        plot_signal(
            'Single-Sided Amplitude Spectrum of Signal Before Processing',
            f, P1, 'Frequency (Hz)', '|P1(f)|'
        )

    if applyFiltering:
        # Normalize the cutoff frequencies
        Wn1 = cutoffFrequency1 / (Fs / 2)  # Normalized cutoff frequency
        Wn2 = cutoffFrequency2 / (Fs / 2)  # Normalized upper cutoff frequency

        # Design the Butterworth filter
        if filterType == 'low':
            b, a = butter(filterOrder, Wn1, btype='low')
        elif filterType == 'high':
            b, a = butter(filterOrder, Wn1, btype='high')
        elif filterType == 'bandpass':
            b, a = butter(filterOrder, [Wn1, Wn2], btype='bandpass')
        elif filterType == 'stop':
            b, a = butter(filterOrder, [Wn1, Wn2], btype='stop')
        else:
            raise ValueError("Invalid filter type. Choose 'low', 'high', 'bandpass', or 'stop'.")

        # Apply the Butterworth filter
        processedSignal = filtfilt(b, a, signal)  # Use filtfilt to avoid phase distortion
    else:
        processedSignal = signal

    if applySmoothing:
        # Apply the selected smoothing method
        if smoothingMethod == 'moving':
            processedSignal = np.convolve(processedSignal, np.ones(movingAverageWindow) / movingAverageWindow, mode='valid')
        elif smoothingMethod == 'exponential':
            processedSignal = np.append(processedSignal[0], exponentialAlpha * processedSignal[1:] + (1 - exponentialAlpha) * processedSignal[:-1])
        elif smoothingMethod == 'savitzky-golay':
            processedSignal = savgol_filter(processedSignal, savitzkyFrameLength, savitzkyOrder)
        else:
            raise ValueError("Invalid smoothing method. Choose 'moving', 'exponential', or 'savitzky-golay'.")

    if check_frequency:
        # Perform the FFT after processing
        Y_processed = np.fft.fft(processedSignal)

        # Compute the two-sided spectrum and then the single-sided spectrum
        P2_processed = np.abs(Y_processed / L)
        P1_processed = P2_processed[:L // 2 + 1]
        P1_processed[1:-1] = 2 * P1_processed[1:-1]

        # Plot the single-sided amplitude spectrum after processing
        plot_signal(
            'Single-Sided Amplitude Spectrum of Signal After Processing',
            f, P1_processed, 'Frequency (Hz)', '|P1(f)|'
        )

    # Plot the original signal
    plot_signal(
        'Original Signal',
        t, signal, 'Time (s)', 'Amplitude'
    )

    # Plot the processed signal (filtered or just smoothed)
    plot_signal(
        'Processed Signal',
        t, processedSignal, 'Time (s)', 'Amplitude'
    )

    # Keep the plots open
    plt.show()

if __name__ == '__main__':
    main()
