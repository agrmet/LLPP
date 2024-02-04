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
int K = 8; // How many threads we will spawn


void Ped::Model::setup(std::vector<Ped::Tagent *> agentsInScenario, std::vector<Twaypoint *> destinationsInScenario, IMPLEMENTATION implementation)
{
	// Convenience test: does CUDA work on this machine?
	cuda_test();
	
	agents = std::vector<Ped::Tagent *>(agentsInScenario.begin(), agentsInScenario.end());
	
	// Set up destinations
	destinations = std::vector<Ped::Twaypoint *>(destinationsInScenario.begin(), destinationsInScenario.end());

	// Sets the chosen implemenation. Standard in the given code is SEQ
	this->implementation = implementation;
	// Set up vectorized agents:
	if (implementation == VECTOR) {
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
		// 2) calculate its next desired position
		agent->computeNextDesiredPosition();
		// 3) set its position to the calculated desired one
		agent->setX(agent->getDesiredX());
		agent->setY(agent->getDesiredY());
	}
}

void Ped::Model::tick()
{
	// Vectorized OpenMP ?
	if (this-> implementation == VECTOR) {
	#pragma omp parallel for num_threads(4) default(none)
	// Compute each agent's next position, all these computations are vectorized:
	for (int i = 0; i < agents.size(); i+=4) {
		this->agentsSoA.computeNextDesiredPositionsVectorized(i);
		// Not sure if we need this
		//this->agentsSoA.updateCoordinatesVectorized(i,agents);
	}
	// Set x and y coordinates for each agent. This part is not vectorized since Tagent is AoS.
	#pragma omp parallel for num_threads(4) default(none)
		for (int i = 0; i < agents.size(); i++){
			agents[i]->setX(this->agentsSoA.xP[i]);
			agents[i]->setY(this->agentsSoA.yP[i]);
		}
	}


	// C++ threads implementation
	if (this->implementation == PTHREAD) {
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
	if (this->implementation == OMP) {
	#pragma omp parallel for num_threads(8) default(none)
	for (Ped::Tagent *agent : agents)
	{
		// 2) calculate its next desired position
		agent->computeNextDesiredPosition();
		// 3) set its position to the calculated desired one
		agent->setX(agent->getDesiredX());
		agent->setY(agent->getDesiredY());
	}
	}

	// Serial implementation
	if (this->implementation == SEQ) {
	for (Ped::Tagent *agent : agents)
	{
		// 2) calculate its next desired position
		agent->computeNextDesiredPosition();
		// 3) set its position to the calculated desired one
		agent->setX(agent->getDesiredX());
		agent->setY(agent->getDesiredY());
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
	// Search for neighboring agents
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

			break;
		}
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

	// create the output list
	// ( It would be better to include only the agents close by, but this programmer is lazy.)
	return set<const Ped::Tagent *>(agents.begin(), agents.end());
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
