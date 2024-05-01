
function writeMatricesToExcel(filename, varargin)
    % This function writes multiple matrices to an Excel file horizontally without overlapping.
    % filename: string representing the name and path of the Excel file
    % varargin: variable input arguments, each expected to be a matrix

    % Initialize the starting column
    startCol = 1;

    % Enhanced function to convert numeric column index to Excel column letters
    function excelCol = colNumToExcelCol(n)
        letters = '';
        while n > 0
            modVal = mod(n-1, 26);
            letters = [char(65 + modVal), letters]; % Concatenate in reverse order
            n = floor((n - modVal) / 26);
        end
        excelCol = letters;
    end

    % Loop through each matrix provided in varargin
    for idx = 1:length(varargin)
        % Current matrix
        currentMatrix = varargin{idx};

        % Write current matrix to Excel
        rangeStr = sprintf('%s1', colNumToExcelCol(startCol)); % Compute the range string
        writematrix(currentMatrix, filename, 'Sheet', 'Sheet1', 'Range', rangeStr);

        % Update startCol for the next matrix
        startCol = startCol + size(currentMatrix, 2) + 1; % Move start column beyond the current matrix
    end
end



