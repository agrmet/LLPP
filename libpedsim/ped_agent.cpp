//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_agent.h"
#include "ped_waypoint.h"
#include <stdio.h>
#include <math.h>
#include <emmintrin.h>


#include <stdlib.h>

// ------------ TagentSoA -------------- 
// New vectorized struct for Tagent (SoA)
Ped::TagentSoA::TagentSoA(std::vector<Tagent*> &agentsInScenario){
	for (int i = 0; i < agentsInScenario.size(); i++){
		this->xP.push_back(agentsInScenario[i]->getX());
		this->yP.push_back(agentsInScenario[i]->getY());
		this->xDesP.push_back(agentsInScenario[i]->getDesiredX());
		this->yDesP.push_back(agentsInScenario[i]->getDesiredY());
		this->waypointsAll.push_back(agentsInScenario[i]->getWaypoints());
	}
	for (int i = 0; i < waypointsAll.size(); i++) {
		this->xWP.push_back(waypointsAll[i].front()->getx());
        this->yWP.push_back(waypointsAll[i].front()->gety());
        this->id.push_back(waypointsAll[i].front()->getid());
        this->r.push_back(waypointsAll[i].front()->getr());
	}
}

void Ped::TagentSoA::getNextDestinationsVectorized(int idx) {
	// booleans to store if these agents reached their destinations (bool* not good need float)
	bool agentsReachedDestination[4];
	// x coordinates
    __m128 x = _mm_load_ps(xP[idx]);
	// y coordinates
	__m128 y = _mm_load_ps(yP[idx]);
	// waypoints x coordinates
	__m128 wpx = _mm_load_ps(xWP[idx]);
	// waypoints y coordinates
	__m128 wpy = _mm_load_ps(yWP[idx]);
	// waypoints radiuses
	__m128 rad = _mm_load_ps(r[idx]);

	// double diffX = destination->getx() - x;
	diffX = _mm_sub_ps(wpx, x);
	// double diffY = destination->gety() - y;
	diffY = _mm_sub_ps(wpy, y);
	// double length = sqrt(diffX * diffX + diffY * diffY);
	__m128 length = _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(diffX, diffX), _mm_mul_ps(diffY, diffY)));
	
	// agentReachedDestination = length < destination->getr();
	_mm_store_ps(agentsReachedDestination, _mm_cmplt_ps(length, rad));

	// loop through the 4 agents
	for (int a = 0; a < 4; a++){
		// Case 1: agent has reached destination (or has no current destination);
		// get next destination if available
		if (agentsReachedDestination[a] && !waypointsAll[idx+a].empty()) {
				// Retrieve next destination from the queue of waypoints:
				// nextDestination = waypoints.front();
				Twaypoint* nextDestination = waypointsAll[idx + a].front(); 
				// Move this destination to the back of the queue:
				// waypoints.push_back(destination);
				// waypoints.pop_front();
				waypointsAll[idx + a].push_back(nextDestination);
				waypointsAll[idx + a].pop_front();

				// Update destination information for this agent:
				id[idx + a] = nextDestination->getid();
				r[idx + a] = nextDestination->getr();
				xWP[idx + a] = nextDestination->getx();
				yWP[idx + a] = nextDestination->gety();
			}
		}
		// else: do nothing. No need to update. 
		// Case 2: agent has not yet reached destination, continue to move towards
		// current destination
	}


void Ped::TagentSoA::computeNextDesiredPositionsVectorized(int idx) {
	getNextDestinationsVectorized(idx);
	printf ("slay");
	return;
}
// ------------- Tagent -----------------

Ped::Tagent::Tagent(int posX, int posY) {
	Ped::Tagent::init(posX, posY);
}

Ped::Tagent::Tagent(double posX, double posY) {
	Ped::Tagent::init((int)round(posX), (int)round(posY));
}

void Ped::Tagent::init(int posX, int posY) {
	x = posX;
	y = posY;
	destination = NULL;
	lastDestination = NULL;
}

void Ped::Tagent::computeNextDesiredPosition() {
	destination = getNextDestination();
	if (destination == NULL) {
		// no destination, no need to
		// compute where to move to
		return;
	}

	double diffX = destination->getx() - x;
	double diffY = destination->gety() - y;
	double len = sqrt(diffX * diffX + diffY * diffY);
	desiredPositionX = (int)round(x + diffX / len);
	desiredPositionY = (int)round(y + diffY / len);
}

void Ped::Tagent::addWaypoint(Twaypoint* wp) {
	waypoints.push_back(wp);
}

Ped::Twaypoint* Ped::Tagent::getNextDestination() {
	Ped::Twaypoint* nextDestination = NULL;
	bool agentReachedDestination = false;

	if (destination != NULL) {
		// compute if agent reached its current destination
		double diffX = destination->getx() - x;
		double diffY = destination->gety() - y;
		double length = sqrt(diffX * diffX + diffY * diffY);
		agentReachedDestination = length < destination->getr();
	}

	if ((agentReachedDestination || destination == NULL) && !waypoints.empty()) {
		// Case 1: agent has reached destination (or has no current destination);
		// get next destination if available
		waypoints.push_back(destination);
		nextDestination = waypoints.front();
		waypoints.pop_front();
	}
	else {
		// Case 2: agent has not yet reached destination, continue to move towards
		// current destination
		nextDestination = destination;
	}

	return nextDestination;
}
