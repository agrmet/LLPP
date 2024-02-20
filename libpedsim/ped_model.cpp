//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_model.h"
#include "ped_waypoint.h"
#include "ped_model.h"
#include "ped_agent.h"
#include <iostream>
#include <stack>
#include <algorithm>
#include "cuda_testkernel.h"
#include <omp.h>
#include <thread>
#include <stdlib.h>
#include <mutex>
#include <math.h>

int K = 4; // How many threads we will spawn
int R = 4; // How many regions we will divide the world into
bool PARALLELMOVE = true; // Whether to move agents in parallel or not
int MAX_X = 160; // Maximum x coordinate of the world (range is 0 to MAX_X)
bool FIRST_TICK = true; // Ensures that the regions are divided at the first tick

// Vector of mutexes for each border between regions
// i.e. 4 regions have 3 borders, so we need 3 mutexes
std::vector<std::mutex> regionMutex(R-1);

// Vector of regions with agents that just changed their region
std::vector<std::vector<Ped::Tagent *>> regionsChangedAgents(R);

// Vector of regions with agents
std::vector<std::vector<Ped::Tagent *>> regions(R);

int getRegionByX(int x)
{
	// Used to determine what region a specific x coordinate belongs to
	if (x < 0) { return 0; }
	if (x >= MAX_X) { return R - 1; }

	// Floor assures div is rounded down
	return floor(x / floor((MAX_X + 1) / R));
}

int getRegionByPosition(Ped::Tagent *agent)
{
	int x = agent->getX();
	return getRegionByX(x);
}

int getCurrentRegion(Ped::Tagent *agent)
{
	int region = agent->getCurrentRegion();

	if (region < 0 || region >= R)
	{
		throw std::out_of_range("Agent's current region is invalid. Region = " + std::to_string(region));
	}
	
	return region;
}

// Divide the 2D world into regions and assign agents to regions
void divideRegions(std::vector<Ped::Tagent *> &agents)
{
	for (Ped::Tagent *agent : agents)
	{
		int region = getRegionByPosition(agent);
		
		regions[region].push_back(agent);
		agent->setCurrentRegion(region);
	}
}

void Ped::Model::setup(std::vector<Ped::Tagent *> agentsInScenario, std::vector<Twaypoint *> destinationsInScenario, IMPLEMENTATION implementation)
{
	// Convenience test: does CUDA work on this machine?
	cuda_test();

	agents = std::vector<Ped::Tagent *>(agentsInScenario.begin(), agentsInScenario.end());

	// Set up destinations
	destinations = std::vector<Ped::Twaypoint *>(destinationsInScenario.begin(), destinationsInScenario.end());

	// Sets the chosen implemenation. Standard in the given code is SEQ
	this->implementation = implementation;

	// Reset the first tick flag in case we are restarting the simulation with a different implementation
	FIRST_TICK = true;

	// Set up vectorized agents:
	if (implementation == VECTOR || implementation == VECTOROMP) {
		this->agentsSoA = TagentSoA(agentsInScenario);
	}

	// Set up heatmap (relevant for Assignment 4)
	setupHeatmapSeq();
}

void agent_tasks(int thread_id, std::vector<Ped::Tagent *> agents)
{
	int N = agents.size();
	int chunkSize = N / K; // Base chunk size
	int remainder = N % K; // Remainder to be distributed

	// Calculate the starting and ending index for this thread
	int start = thread_id * chunkSize + std::min(thread_id, remainder);
	int end = start + chunkSize + (thread_id < remainder ? 1 : 0);

	for (int i = start; i < end; i++)
	{
		Ped::Tagent *agent = agents[i];
		// Calculate its next desired position
		agent->computeNextDesiredPosition();
	}
}

void Ped::Model::tick()
{
	if (FIRST_TICK && PARALLELMOVE && this->implementation != VECTOR && this->implementation != VECTOROMP) {
		// Clear the regions and regionsChangedAgents vectors
		for (int i = 0; i < R; i++) {
			regions[i].clear();
			regionsChangedAgents[i].clear();
		}
		// Unlock all the region mutexes
		for (int i = 0; i < (R-1); i++) {
			regionMutex[i].unlock();
		}

		// Divide the 2D world into regions and assign agents to regions
		divideRegions(agents);
		
		FIRST_TICK = false;
	}

	// Vectorized ONLY
	if (this->implementation == VECTOR)
	{
		// Compute each agent's next position, all these computations are vectorized:
		for (int i = 0; i < agents.size(); i += 4)
		{
			this->agentsSoA.computeNextPositionsVectorized(i);
		}
		// Set x and y coordinates for each agent. This part is not vectorized since Tagent is AoS.
		for (int i = 0; i < agents.size(); i++)
		{
			agents[i]->setX(this->agentsSoA.xP[i]);
			agents[i]->setY(this->agentsSoA.yP[i]);
		}
	}

	// Vectorized WITH OMP
	if (this->implementation == VECTOROMP)
	{
		#pragma omp parallel for num_threads(K) default(none)
		// Compute each agent's next position, all these computations are vectorized:
		for (int i = 0; i < agents.size(); i += 4)
		{
			this->agentsSoA.computeNextPositionsVectorized(i);
		}
		
		#pragma omp parallel for num_threads(K) default(none)
		// Set x and y coordinates for each agent. This part is not vectorized since Tagent is AoS.
		for (int i = 0; i < agents.size(); i++)
		{
			agents[i]->setX(this->agentsSoA.xP[i]);
			agents[i]->setY(this->agentsSoA.yP[i]);
		}
	}

	// C++ threads implementation
	if (this->implementation == PTHREAD)
	{
		std::thread threads[K];
		for (int i = 0; i < K; i++)
		{
			threads[i] = std::thread(agent_tasks, i, agents);
		}
		for (int i = 0; i < K; i++)
		{
			threads[i].join();
		}
	}

	// OpenMP implementation
	if (this->implementation == OMP)
	{
	#pragma omp parallel for num_threads(K) default(none)
		for (Ped::Tagent *agent : agents)
		{
			// Calculate its next desired position
			agent->computeNextDesiredPosition();
		}
	}

	// Serial implementation
	if (this->implementation == SEQ)
	{
		for (Ped::Tagent *agent : agents)
		{
			// Calculate its next desired position
			agent->computeNextDesiredPosition();
		}
	}

	if (this->implementation != VECTOR && this->implementation != VECTOROMP) {
		if (PARALLELMOVE) { 
			// Move agents in parallel with OpenMP

			#pragma omp parallel for num_threads(K) default(none) shared(K, regions, regionsChangedAgents, regionMutex)
			for (int i = 0; i < K; i++)
			{
				for (Ped::Tagent *agent : regions[i])
				{
					move(agent);
				}
			}

			// Handle agents changing regions sequentially
			for (int i = 0; i < K; i++)
			{
				for (Ped::Tagent *agent : regionsChangedAgents[i])
				{
					int new_region = getRegionByPosition(agent);
					int old_region = i;

					// Remove the agent from the old region
					regions[old_region].erase(std::remove(regions[old_region].begin(), regions[old_region].end(), agent), regions[old_region].end());
					// Change the agent's current region
					agent->setCurrentRegion(new_region);
					// Insert the agent into the new region
					regions[new_region].push_back(agent);					
				}
				// Clear the list of agents that changed regions
				regionsChangedAgents[i].clear();
			}
		}
		else {
			// Move agents sequentially
			for (Ped::Tagent *agent : agents)
			{
				move(agent);
			}
		}
	}
}

////////////
/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////

// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.
void Ped::Model::move(Ped::Tagent *agent)
{
	int currentRegion = -1;
	bool leftBorder = false;
	bool rightBorder = false;

	if (PARALLELMOVE) {
		currentRegion = agent->getCurrentRegion();
		int x = agent->getX();

		// Check if the agent is within dist 2 of a region border
		leftBorder = getRegionByX(x - 2) != currentRegion;
		rightBorder = getRegionByX(x + 2) != currentRegion;

		// Lock the mutex of the border between the regions
		if (leftBorder) { regionMutex[currentRegion - 1].lock(); }
		if (rightBorder) {	regionMutex[currentRegion].lock();	}
	}

	// Search for neighboring agents (within dist 2)
	set<const Ped::Tagent *> neighbors = getNeighbors(agent->getX(), agent->getY(), 2);

	// Retrieve their positions
	std::vector<std::pair<int, int>> takenPositions;
	for (std::set<const Ped::Tagent *>::iterator neighborIt = neighbors.begin(); neighborIt != neighbors.end(); ++neighborIt)
	{
		std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
		takenPositions.push_back(position);
	}

	// Compute the three alternative positions that would bring the agent
	// closer to his desiredPosition, starting with the desiredPosition itself
	std::vector<std::pair<int, int>> prioritizedAlternatives;
	std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
	prioritizedAlternatives.push_back(pDesired);

	int diffX = pDesired.first - agent->getX();
	int diffY = pDesired.second - agent->getY();
	std::pair<int, int> p1, p2;
	if (diffX == 0 || diffY == 0)
	{
		// Agent wants to walk straight to North, South, West or East
		p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
		p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
	}
	else
	{
		// Agent wants to walk diagonally
		p1 = std::make_pair(pDesired.first, agent->getY());
		p2 = std::make_pair(agent->getX(), pDesired.second);
	}
	prioritizedAlternatives.push_back(p1);
	prioritizedAlternatives.push_back(p2);

	// Find the first empty alternative position
	for (std::vector<pair<int, int>>::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it)
	{
		// If the current position is not yet taken by any neighbor
		if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end())
		{

			// Set the agent's position
			agent->setX((*it).first);
			agent->setY((*it).second);

			if (PARALLELMOVE) {
				int new_region = getRegionByPosition(agent);
				int old_region = getCurrentRegion(agent);

				if (new_region != old_region)
				{
					// Add the agent to the list of agents that changed regions
					regionsChangedAgents[old_region].push_back(agent);
				}
			}

			break;
		}
	}

	if (PARALLELMOVE) {
		// Unlock the mutex of the border between the regions
		if (rightBorder) { regionMutex[currentRegion].unlock(); }
		if (leftBorder) { regionMutex[currentRegion - 1].unlock(); }
	}
}

/// Returns the list of neighbors within dist of the point x/y. This
/// can be the position of an agent, but it is not limited to this.
/// \date    2012-01-29
/// \return  The list of neighbors
/// \param   x the x coordinate
/// \param   y the y coordinate
/// \param   dist the distance around x/y that will be searched for agents (search field is a square in the current implementation)
set<const Ped::Tagent *> Ped::Model::getNeighbors(int x, int y, int dist) const
{
	// The set of neighbors to be returned (only if they are within dist)
	set<const Ped::Tagent *> neighbors;

	// For each agent, check if it's within dist
	for (std::vector<Ped::Tagent *>::const_iterator agentIt = agents.begin(); agentIt != agents.end(); ++agentIt)
	{
		int dx = (*agentIt)->getX() - x;
		int dy = (*agentIt)->getY() - y;
		if (sqrt(dx * dx + dy * dy) < dist)
		{
			// Add to the list only if within dist
			neighbors.insert(*agentIt);
		}
	}

	return neighbors;
}

void Ped::Model::cleanup()
{
	// Nothing to do here right now.
}

Ped::Model::~Model()
{
	std::for_each(agents.begin(), agents.end(), [](Ped::Tagent *agent)
				  { delete agent; });
	std::for_each(destinations.begin(), destinations.end(), [](Ped::Twaypoint *destination)
				  { delete destination; });
}
