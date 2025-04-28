function y=butterfilterlpf(data,cutoff,fs,order)
% 4th order, zero-phase lowpass filter
%
% Dirk Weenk, 2011-09-20
%
if nargin~=4
    error('4 inputs required!')
end
[b,a]=butter(order,cutoff/(fs/2));
y=filtfilt(b,a,data);