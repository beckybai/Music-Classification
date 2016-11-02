function [s,s2] = example(musicname)

%load a .wav file
[x, fs] = wavread(musicname);     % get the samples of the .wav file
x = x(:, 2);                        % get the first channel
xmax = max(abs(x));                 % find the maximum abs value
x = x/xmax;                         % scalling the signal

% define analysis parameters
xlen = length(x);                   % length of the signal
wlen = 512*4;                        % window length (recomended to be power of 2)
h = wlen/2;                         % hop size (recomended to be power of 2)
nfft = 2048;                        % number of fft points (recomended to be power of 2)

% define the coherent amplification of the window
K = sum(hamming(wlen, 'periodic'))/wlen;

% perform STFT
[s, f, t] = stft(x, wlen, h, nfft, fs);

% take the amplitude of fft(x) and scale it, so not to be a
% function of the length of the window and its coherent amplification
s = abs(s)/wlen/K;

% correction of the DC & Nyquist component
if rem(nfft, 2)                     % odd nfft excludes Nyquist point
    s(2:end, :) = s(2:end, :).*2;
else                                % even nfft includes Nyquist point
    s(2:end-1, :) = s(2:end-1, :).*2;
end

% convert amplitude spectrum to dB (min = -120 dB)
s2 = 20*log10(s + 1e-6);
s = (-s2);

% % plot the spectrogram
% figure
% imagesc(t, f, s)
% set(gca,'YDir','normal')
% set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
% xlabel('Time, s')
% ylabel('Frequency, Hz')
% title('Amplitude spectrogram of the signal')
% 
% handl = colorbar;
% set(handl, 'FontName', 'Times New Roman', 'FontSize', 14)
% ylabel(handl, 'Magnitude, dB')