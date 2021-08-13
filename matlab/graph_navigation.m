%
% Results of the experiments.
%

function MDP = maze_navigation
rng('default')
label.factor     = {'where'};
label.modality   = {'distance'};

GRAPH = [...
    1 1 1 1 1 1 1 1 1 1 1 1;
    1 1 0 0 0 0 0 0 0 1 1 1;
    1 0 1 1 1 1 1 1 1 1 1 1;
    1 1 0 0 0 0 0 0 0 0 0 1;
    1 1 1 1 1 1 1 1 1 1 1 1;
    1 0 1 1 1 1 1 1 1 1 1 1;
    1 1 1 1 1 1 1 1 1 1 1 1;
];

NB_GOOD_PATHS = 2;
NB_BAD_PATHS = 5;
GOOD_PATH_1 = [3 4 5 6 7 8 9];              % states of the first path
GOOD_PATH_2 = [10 11 12 13 14 15 16 17 18]; % states of the second path

START_STATE = 1;
EXIT_STATE  = 18;
STATES      = 18;    % Number of states
OUTCOMES    = 2;     % Number of outcomes
ACTIONS     = NB_GOOD_PATHS + NB_BAD_PATHS ; % Number of actions
NOISE       = 0.2;
SIGNAL      = 1 - NOISE;
TRIALS      = 20;
C_ALPHA     = 8; % Scaling factor for prior preferences

% prior beliefs about initial states: D 
%--------------------------------------------------------------------------
D{1} = ones(STATES,1) * NOISE / (STATES - 1);
D{1}(1,1) = SIGNAL;

% probabilistic mapping from hidden states to outcomes: A
%--------------------------------------------------------------------------
A{1} = ones(OUTCOMES,STATES) * NOISE / (OUTCOMES - 1);
for s = 1:STATES
    if (s ~= 2)
        A{1}(1, s) = SIGNAL;
    else
        A{1}(2, s) = SIGNAL;
    end
end

% controlled transitions: B
%--------------------------------------------------------------------------
B{1} = zeros(STATES,STATES,ACTIONS);
for si = 1:STATES
    for ai = 1:ACTIONS
        ss = simlate_action(si,ai,GOOD_PATH_1,GOOD_PATH_2);
        B{1}(ss,si,ai) = B{1}(ss,si,ai) + SIGNAL;
        for p = 1:ACTIONS
            if (p ~= ai)
                ss = simlate_action(si,p,GOOD_PATH_1,GOOD_PATH_2);
                B{1}(ss,si,ai) = B{1}(ss,si,ai) + NOISE / (ACTIONS - 1);
            end
        end
    end
end
disp(B{1})

% allowable policies (8 moves): V
%--------------------------------------------------------------------------
V     = [];
for i1 = 1:ACTIONS
for i2 = 1:ACTIONS
for i3 = 1:ACTIONS
for i4 = 1:ACTIONS
for i5 = 1:ACTIONS
for i6 = 1:ACTIONS
for i7 = 1:ACTIONS
for i8 = 1:ACTIONS % Crashes ...
    V(:,end + 1) = [i1;i2;i3;i4;i5;i6;i7;i8];
end
end
end
end
end
end
end
end

% priors: (negative cost) C:
%--------------------------------------------------------------------------
C{1} = zeros(OUTCOMES,1);
for g = 1:OUTCOMES
    C{1}(g) = C_ALPHA * (OUTCOMES - g);
end

% basic MDP structure
%--------------------------------------------------------------------------
mdp.V = V;                      % allowable policies
mdp.A = A;                      % observation model or likelihood
mdp.B = B;                      % transition probabilities
mdp.C = C;                      % preferred outcomes
mdp.D = D;                      % prior over initial states

mdp.label = label;
mdp       = spm_MDP_check(mdp);

% exploratory sequence (with experience and task set)
%==========================================================================
tic
MDP = spm_maze_search(mdp,TRIALS,START_STATE,EXIT_STATE,128,1);
toc

% show results in terms of path
%--------------------------------------------------------------------------
spm_figure('GetWin','Figure 1'); clf
spm_maze_plot(MDP,GRAPH,GOOD_PATH_1,GOOD_PATH_2);
end

function MDP = spm_maze_search(mdp,N,START,END,alpha,beta)
% FORMAT MDP = spm_maze_search(mdp,N,START,END,alpha,beta)
% mdp   - MDP structure
% N     - number of trials (i.e., policies: default 8)
% START - index of intial state (default 1)
% END   - index of target state (default 1)
% alpha - prior concentration parameter for likelihood (default 128)
% beta  - precision of prior preference (default 0)
%the argument is
% MDP   - MDP structure array

% preliminaries
%--------------------------------------------------------------------------
try, N;     catch, N     = 8;   end
try, START; catch, START = 1;   end
try, END;   catch, END   = 1;   end
try, alpha; catch, alpha = 128; end
try, beta;  catch, beta  = 0;   end

mdp.s = START;

% Evaluate a sequence of moves
%==========================================================================
for i = 1:N
    % proceed with subsequent trial
    %----------------------------------------------------------------------
    disp("Trial number:");
    disp(i);
    MDP(i)   = spm_MDP_VB_X(mdp);
    mdp      = MDP(i);
    mdp.s    = mdp.s(:,end);
    mdp.D{1} = MDP(i).X{1}(:,end);
    mdp.o    = [];
    mdp.u    = [];
end
end

function ss = simlate_action(s, a, path_1, path_2)
    % Get size of the longest path
    longest_path_size = size(path_1,2);
    if (longest_path_size < size(path_2,2))
        longest_path_size = size(path_2,2);
    end
    % If initial state and good action selected.
    if (s == 1 && a <= 2)
        % Go to the first state of the path corresponding to the "action".
        if (a == 1)
            ss = path_1(1);
        else
            ss = path_2(1);
        end
        return;
    end
    % If agent is in initial state and selected a bad action or the agent is in the bad state already.
    if (s == 1 || s == 2)
        % Go to bad state.
        ss = 2;
        return;
    end
    % If already engaged in a path and select bad action.
    if (a > 1)
        % Go to bad state.
        ss = 2;
        return;
    end
    % If already engaged in a path 1 and select good action.
    for i = 1:size(path_1,2)
        if (s == path_1(i))
            if (i ~= size(path_1,2))
                ss = path_1(i + 1);
                return;
            else
                if (longest_path_size == size(path_1,2))
                    ss = path_1(i);
                else
                    ss = 2;
                end
                return;
            end
        end
    end
    % If already engaged in a path 2 and select good action.
    for i = 1:size(path_2,2)
        if (s == path_2(i))
            if (i ~= size(path_2,2))
                ss = path_2(i + 1);
                return;
            else
                if (longest_path_size == size(path_2,2))
                    ss = path_2(i);
                else
                    ss = 2;
                end
                return;
            end
        end
    end
    throw(MException('Invalid action or state, when calling simlate_action(...)'));
end

function res = invert_color(GRAPH)
res = zeros(size(GRAPH));
for y = 1:size(GRAPH,1)
    for x = 1:size(GRAPH,2)
        if (GRAPH(y,x) == 0)
            res(y,x) = 1;
        end
    end
end
end

function [yp,xp] = state_to_position(path_1,path_2,state)
    % Agent is in initial state
    if (state == 1)
        yp = 3;
        xp = 2;
        return;
    end
    
    % Agent is in bad state
    if (state == 2)
        yp = 6;
        xp = 2;
        return;
    end
    
    % Agent is in first path
    for i = 1:size(path_1,2)
        if (s == path_1(i))
            yp = 2;
            xp = 2 + i;
            return;
        end
    end
    
    % Agent is in second path
    for i = 1:size(path_2,2)
        if (s == path_2(i))
            yp = 4;
            xp = 2 + i;
            return;
        end
    end
    
end

function spm_maze_plot(MDP,GRAPH,GOOD_PATH_1,GOOD_PATH_2)
% display maze
%--------------------------------------------------------------------------
MAZE = invert_color(GRAPH);
subplot(2,2,1), imagesc(MAZE), axis image
title('Scanpath','fontsize',16);

% Cycle of the trials
%--------------------------------------------------------------------------
h     = [];
MS    = {};
MC    = {};
for p = 1:numel(MDP)
    %  display prior preferences
    %----------------------------------------------------------------------
    C     = MDP(p).C{1}(:,1);
    C     = spm_softmax(C);
    subplot(2,2,3), imagesc(C), axis image
    title('Preferences','fontsize',16);
    
    % cycle over  short-term searches
    %----------------------------------------------------------------------
    subplot(2,2,1),hold on
    s     = MDP(p).s;
    for t = 1:numel(s)
        % location
        %------------------------------------------------------------------
        [i,j] = state_to_position(GOOD_PATH_1,GOOD_PATH_2,s(t));
        h(end + 1) = plot(j,i,'.','MarkerSize',32,'Color','r');
        try
            set(h(end - 1),'Color','m','MarkerSize',16);
            j = [get(h(end - 1),'Xdata'), get(h(end),'Xdata')];
            i = [get(h(end - 1),'Ydata'), get(h(end),'Ydata')];
            plot(j,i,':r');
        end
        
        % save
        %------------------------------------------------------------------
        if numel(MS)
            MS(end + 1) = getframe(gca);
        else
            MS = getframe(gca);
        end
        
    end
    % save
    %----------------------------------------------------------------------
    subplot(2,2,3)
    if numel(MC)
        MC(end + 1) = getframe(gca);
    else
        MC = getframe(gca);
    end
    
end

% save movie
%--------------------------------------------------------------------------
subplot(2,2,1)
xlabel('click axis for movie')
set(gca,'Userdata',{MS,16})
set(gca,'ButtonDownFcn','spm_DEM_ButtonDownFcn')

subplot(2,2,3)
xlabel('click axis for movie')
set(gca,'Userdata',{MC,16})
set(gca,'ButtonDownFcn','spm_DEM_ButtonDownFcn')
end