function options = processOptions(optionsInput, optionsDefault)
% processOptions is a helper function for preprocessImage.m
%
% processOptions was provided by Patrick Pletscher in the HOL package:
% https://github.com/ppletscher/hol

options = optionsDefault;
if (isstruct(optionsInput))
    fields = fieldnames(optionsInput);
    for i=1:length(fields)
        if (~isstruct(getfield(optionsInput, fields{i})))
            options = setfield(options, fields{i}, getfield(optionsInput, fields{i}));
        else
            temp = processOptions(getfield(optionsInput, fields{i}), ...
                            getfield(options, fields{i}) );
            options = setfield(options, fields{i}, temp);
        end
    end
end

end % processOptions
