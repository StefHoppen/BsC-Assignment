function data = readCalibratedFSMTxLog(filename)
% Henk Kortier Jan 2012
% Dirk Weenk, 2013-03-21 changed to read MTx files, exported with MT manager ForceShoe

fid = fopen(filename);

if fid == -1
    disp('invalid mtw file');
    
else
    
    data.StartTime = str2double(regexp(fgetl(fid),'[0-9]+','match'));
    data.fs = str2double(regexp(fgetl(fid),'[0-9]+\.[0-9]+','match'));
    data.Scenario = regexp(fgetl(fid),'[0-9]+\.[0-9]','match');
    data.FW = regexp(fgetl(fid),'[0-9]+\.[0-9]+\.[0-9]+','match');
    data.UnloadedForceShoe = fgetl(fid);
    
    tline = fgetl(fid);
    while isempty(regexp(tline,'Counter', 'once'))
        tline = fgetl(fid);
        header = regexp(tline,'\w+\s\w+\:','match');
        
        %    if (strcmp(header,'Sample rate:'))
        data = regexp(tline,'(?<fs>\d+\.\d+)','names');
        %    end
    end
    
    header = regexp(tline,'\t','split');
    numcols = length(header)-1;
    
    values = textscan(fid, repmat('%f',1,numcols), 'delimiter', '\t');%, 'HeaderLines',4);
    
    fclose(fid);
    
    for i = 1:length(values)
        %Remove first 0.25s at start and end samples
        ncut = round(0.25*data.fs);
        values{i} = values{i}(ncut:end-ncut);
        
        isnanValues = isnan(values{i});
        if sum(isnanValues)>0
            values{i}(isnanValues) = 0;
            %disp('NaN values set to zero
        end
    end
    
    % find Counter column
    data.counter = values{strcmp(header,'Counter')};
    data.N = length(data.counter);
    data.header{1} = 'counter';
    
    for i = 1:numcols
        if strcmp(header{i},'Acc_X')
            data.acc = [values{i} values{i+1} values{i+2}];
            data.header{end+1} = 'acc';
        end
        
        if strcmp(header{i},'Gyr_X')
            data.gyr = [values{i} values{i+1} values{i+2}];
            data.header{end+1} = 'gyr';
        end
        
        if strcmp(header{i},'Mag_X')
            data.mag = [values{i} values{i+1} values{i+2}];
            data.header{end+1} = 'mag';
        end
        
        if strcmp(header{i},'Quat_w')
            data.qgs = [values{i} values{i+1} values{i+2} values{i+3}];
            data.header{end+1} = 'qgs';
        end
        
        if strcmp(header{i},'Mat[0][0]')
%             warning('check if first column or rows')
            data.RgsXsens=zeros(data.N,3,3);
                        data.RgsXsens(:,:,1)=[values{i} values{i+1} values{i+2} ];
                        data.RgsXsens(:,:,2)=[values{i+3} values{i+4} values{i+5} ];
                        data.RgsXsens(:,:,3)=[values{i+6} values{i+7} values{i+8} ];
%             data.RgsXsens(:,1,:)=[values{i} values{i+1} values{i+2} ];
%             data.RgsXsens(:,2,:)=[values{i+3} values{i+4} values{i+5} ];
%             data.RgsXsens(:,3,:)=[values{i+6} values{i+7} values{i+8} ];
            data.header{end+1} = 'RgsXsens';
        end
        
        if strcmp(header{i},'OriInc_w')
            data.dqgs = [values{i} values{i+1} values{i+2} values{i+3}];
            data.header{end+1} = 'dqgs';
        end
        
        if strcmp(header{i},'VelInc_X')
            data.dv = [values{i} values{i+1} values{i+2}];
            data.header{end+1} = 'dv';
        end
        
        if strcmp(header{i},'Temperature')
            data.temp = values{i};
            data.header{end+1} = 'temp';
        end
        
        if strcmp(header{i},'Pressure')
            data.pres = values{i};
            data.header{end+1} = 'pres';
        end
        
        if strcmp(header{i},'Year')
            data.year = values{i};
            data.header{end+1} = 'year';
        end
        
        
        if strcmp(header{i},'Month')
            data.month = values{i};
            data.header{end+1} = 'month';
        end
        
        if strcmp(header{i},'Day')
            data.day = values{i};
            data.header{end+1} = 'day';
        end
        
        if strcmp(header{i},'Second')
            data.second = values{i};
            data.header{end+1} = 'second';
        end
        
        if strcmp(header{i},'AnalogIn_1')
            data.AnalogIn_1 = values{i};
            data.header{end+1} = 'AnalogIn_1';
        end
        
        if strcmp(header{i},'AnalogIn_2')
            data.AnalogIn_2 = values{i};
            data.header{end+1} = 'AnalogIn_2';
        end
        
        if strcmp(header{i},'G0')
            data.G = [values{i} values{i+1} values{i+2} values{i+3} values{i+4} values{i+5} values{i+6}];
            data.header{end+1} = 'G';
        end
        
        if strcmp(header{i},'ref')
            data.ref = values{i};
            data.header{end+1} = 'ref';
        end
        
        if strcmp(header{i},'Fx')
            data.F = [values{i} values{i+1} values{i+2}];
            data.header{end+1} = 'F';
        end
        
        if strcmp(header{i},'Tx')
            data.T = [values{i} values{i+1} values{i+2}];
            data.header{end+1} = 'T';
        end
        
        if strcmp(header{i},'Latitude')
            data.Latitude = values{i};
            data.header{end+1} = 'Latitude';
        end
        
        if strcmp(header{i},'Longitude')
            data.Longitude = values{i};
            data.header{end+1} = 'Longitude';
        end
        
        if strcmp(header{i},'Altitude')
            data.Altitude = values{i};
            data.header{end+1} = 'Altitude';
        end
        
        if strcmp(header{i},'Vel_x')
            data.Vel_gps = [values{i} values{i+1} values{i+2}];
            data.header{end+1} = 'Vel_gps';
        end
        
        if strcmp(header{i},'Status')
            data.Status = values{i};
            data.header{end+1} = 'Status';
        end
    end
end