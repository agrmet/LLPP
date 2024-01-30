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
#include <smmintrin.h>


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
	// floats to store if these agents reached their destinations
	float agentsReachedDestination[4];
	// x coordinates
    __m128 x = _mm_load_ps(&xP[idx]);
	// y coordinates
	__m128 y = _mm_load_ps(&yP[idx]);
	// waypoints x coordinates
	__m128 wpx = _mm_load_ps(&xWP[idx]);
	// waypoints y coordinates
	__m128 wpy = _mm_load_ps(&yWP[idx]);
	// waypoints radiuses
	__m128 rad = _mm_load_ps(&r[idx]);
	// find distance to destination
	__m128 diffX = _mm_sub_ps(wpx, x);
	__m128 diffY = _mm_sub_ps(wpy, y);
	__m128 length = _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(diffX, diffX), _mm_mul_ps(diffY, diffY)));	

	_mm_store_ps(agentsReachedDestination, _mm_cmplt_ps(length, rad));

	// loop through the 4 agents
	for (int a = 0; a < 4; a++){
		// Case 1: agent has reached destination (or has no current destination);
		// get next destination if available
		if (agentsReachedDestination[a] && !waypointsAll[idx+a].empty()) {
				// waypointsAll holds all waypoint including current at top
				Twaypoint* destination = waypointsAll[idx + a].front();
				// pop and reinsert current destination
				waypointsAll[idx + a].pop_front();
				waypointsAll[idx + a].push_back(destination);
				// checkout nextDestination
				Twaypoint* nextDestination = waypointsAll[idx + a].front(); 

				// Update destination information for this agent:
				id[idx + a] = nextDestination->getid();
				r[idx + a] = nextDestination->getr();
				xWP[idx + a] = nextDestination->getx();
				yWP[idx + a] = nextDestination->gety();

				printf("<Agent [%d]> New dest=(%lf,%lf), Current pos=(%lf,%lf)\n", idx+a,xWP[idx+a],yWP[idx+a],xP[idx+a],yP[idx+a]);
			}
		}
		// else: do nothing. No need to update. 
		// Case 2: agent has not yet reached destination, continue to move towards
		// current destination
	}


void Ped::TagentSoA::computeNextDesiredPositionsVectorized(int idx) {
	getNextDestinationsVectorized(idx);
	
	// TODO: Repeated loads/calc of x,y,wpx,wpy,diffX,diffY,length. Merge with getNextDestinationsVectorized?
	// x coordinates
    __m128 x = _mm_load_ps(&xP[idx]);
	// y coordinates
	__m128 y = _mm_load_ps(&yP[idx]);
	// waypoints x coordinates
	__m128 wpx = _mm_load_ps(&xWP[idx]);
	// waypoints y coordinates
	__m128 wpy = _mm_load_ps(&yWP[idx]);
	// find distance to destination
	__m128 diffX = _mm_sub_ps(wpx, x);
	__m128 diffY = _mm_sub_ps(wpy, y);
	__m128 length = _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(diffX, diffX), _mm_mul_ps(diffY, diffY)));	

	// calculate next pos on its way towards destination
	__m128 desX = _mm_add_ps(x, _mm_div_ps(diffX, length));
	__m128 desY = _mm_add_ps(y, _mm_div_ps(diffY, length));
	// update struct
	_mm_store_ps(&xDesP[idx], _mm_round_ps(desX, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)));
	_mm_store_ps(&yDesP[idx], _mm_round_ps(desY, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)));

	return;
}

void Ped::TagentSoA::updateCoordinatesVectorized(int idx, std::vector<Tagent*> &agentsInScenario) {
	// Load desired (x,y) coordinates
    __m128 desX = _mm_load_ps(&xDesP[idx]);
	__m128 desY = _mm_load_ps(&yDesP[idx]);

	// Store new (x,y) coordinates
	_mm_store_ps(&xP[idx], desX);
	_mm_store_ps(&yP[idx], desY);

	// Update Agents-struct for plotting
	// TODO: Sometimes SEGFAULT (probably when not div by 4), updated position doesn't show on graphics. Why?
	for (int a = 0; a < 4; a++){
		// agentsInScenario[idx + a]->setX(xP[idx + a]);
		// agentsInScenario[idx + a]->setY(yP[idx + a]);
	}
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
