#!/bin/bash
#
# Real-time Connection Monitor for Gateway Port 4002
# Shows all connections and their states with timestamps
#

PORT="${1:-4002}"
INTERVAL="${2:-2}"

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

clear

echo -e "${BOLD}${CYAN}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║           Gateway Connection Monitor - Port $PORT                            ║${NC}"
echo -e "${BOLD}${CYAN}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # Get all connections on port
    CONNECTIONS=$(lsof -nP -iTCP:$PORT 2>/dev/null)

    # Count by state
    LISTEN_COUNT=$(echo "$CONNECTIONS" | grep "LISTEN" | wc -l | tr -d ' ')
    ESTABLISHED_COUNT=$(echo "$CONNECTIONS" | grep "ESTABLISHED" | wc -l | tr -d ' ')
    CLOSE_WAIT_COUNT=$(echo "$CONNECTIONS" | grep "CLOSE_WAIT" | wc -l | tr -d ' ')
    TIME_WAIT_COUNT=$(echo "$CONNECTIONS" | grep "TIME_WAIT" | wc -l | tr -d ' ')
    SYN_SENT_COUNT=$(echo "$CONNECTIONS" | grep "SYN_SENT" | wc -l | tr -d ' ')
    CLOSED_COUNT=$(echo "$CONNECTIONS" | grep "CLOSED" | wc -l | tr -d ' ')

    TOTAL_COUNT=$((LISTEN_COUNT + ESTABLISHED_COUNT + CLOSE_WAIT_COUNT + TIME_WAIT_COUNT + SYN_SENT_COUNT + CLOSED_COUNT))

    # Move cursor to top (except first iteration)
    if [ "$FIRST_RUN" != "1" ]; then
        tput cup 5 0
    fi
    FIRST_RUN="0"

    echo -e "${BOLD}[${TIMESTAMP}]${NC} Monitoring Gateway Port $PORT..."
    echo ""

    # Summary Box
    echo -e "${BOLD}┌─────────────────────────────────────────────────────────────────────────┐${NC}"
    echo -e "${BOLD}│ CONNECTION SUMMARY                                                      │${NC}"
    echo -e "${BOLD}├─────────────────────────────────────────────────────────────────────────┤${NC}"
    printf "${BOLD}│${NC} %-20s ${GREEN}%3d${NC} %-49s${BOLD}│${NC}\n" "LISTEN:" "$LISTEN_COUNT" "(Gateway listening for connections)"
    printf "${BOLD}│${NC} %-20s ${GREEN}%3d${NC} %-49s${BOLD}│${NC}\n" "ESTABLISHED:" "$ESTABLISHED_COUNT" "(Active API connections)"
    printf "${BOLD}│${NC} %-20s ${RED}%3d${NC} %-49s${BOLD}│${NC}\n" "CLOSE_WAIT:" "$CLOSE_WAIT_COUNT" "(🧟 ZOMBIES - block new connections!)"
    printf "${BOLD}│${NC} %-20s ${YELLOW}%3d${NC} %-49s${BOLD}│${NC}\n" "TIME_WAIT:" "$TIME_WAIT_COUNT" "(Closing connections)"
    printf "${BOLD}│${NC} %-20s ${CYAN}%3d${NC} %-49s${BOLD}│${NC}\n" "SYN_SENT:" "$SYN_SENT_COUNT" "(Connection attempts in progress)"
    printf "${BOLD}│${NC} %-20s ${MAGENTA}%3d${NC} %-49s${BOLD}│${NC}\n" "CLOSED:" "$CLOSED_COUNT" "(Recently closed)"
    echo -e "${BOLD}├─────────────────────────────────────────────────────────────────────────┤${NC}"
    printf "${BOLD}│${NC} %-20s ${BOLD}%3d${NC} %-49s${BOLD}│${NC}\n" "TOTAL:" "$TOTAL_COUNT" ""
    echo -e "${BOLD}└─────────────────────────────────────────────────────────────────────────┘${NC}"
    echo ""

    # Zombie Alert
    if [ "$CLOSE_WAIT_COUNT" -gt 0 ]; then
        echo -e "${RED}${BOLD}⚠️  WARNING: $CLOSE_WAIT_COUNT ZOMBIE CONNECTION(S) DETECTED!${NC}"
        echo -e "${RED}   Zombies block new API connections and cause timeouts.${NC}"
        echo -e "${RED}   Solution: Restart Gateway to clear zombies.${NC}"
        echo ""
    fi

    # Active Connections Detail
    if [ "$TOTAL_COUNT" -gt 0 ]; then
        echo -e "${BOLD}┌─────────────────────────────────────────────────────────────────────────┐${NC}"
        echo -e "${BOLD}│ ACTIVE CONNECTIONS DETAIL                                               │${NC}"
        echo -e "${BOLD}├─────────────────────────────────────────────────────────────────────────┤${NC}"

        # Header
        printf "${BOLD}│${NC} %-12s %-8s %-12s %-38s ${BOLD}│${NC}\n" "PROCESS" "PID" "STATE" "CONNECTION"
        echo -e "${BOLD}├─────────────────────────────────────────────────────────────────────────┤${NC}"

        # Show each connection
        echo "$CONNECTIONS" | tail -n +2 | while read -r line; do
            COMMAND=$(echo "$line" | awk '{print $1}')
            PID=$(echo "$line" | awk '{print $2}')
            STATE=$(echo "$line" | awk '{print $NF}')
            NODE=$(echo "$line" | awk '{for(i=9;i<NF;i++) printf $i" "; print ""}' | sed 's/  */ /g')

            # Truncate if too long
            COMMAND=$(printf "%.12s" "$COMMAND")
            NODE=$(printf "%.38s" "$NODE")

            # Color based on state
            case "$STATE" in
                "LISTEN")
                    COLOR=$GREEN
                    ;;
                "ESTABLISHED")
                    COLOR=$GREEN
                    ;;
                "CLOSE_WAIT")
                    COLOR=$RED
                    COMMAND="${COMMAND}🧟"
                    ;;
                "TIME_WAIT")
                    COLOR=$YELLOW
                    ;;
                "SYN_SENT")
                    COLOR=$CYAN
                    ;;
                *)
                    COLOR=$NC
                    ;;
            esac

            printf "${BOLD}│${NC} ${COLOR}%-12s %-8s %-12s${NC} %-38s ${BOLD}│${NC}\n" "$COMMAND" "$PID" "$STATE" "$NODE"
        done

        echo -e "${BOLD}└─────────────────────────────────────────────────────────────────────────┘${NC}"
        echo ""
    else
        echo -e "${YELLOW}No connections found on port $PORT${NC}"
        echo ""
    fi

    # Status indicator
    if [ "$CLOSE_WAIT_COUNT" -gt 0 ]; then
        STATUS="${RED}●${NC} ZOMBIES PRESENT"
    elif [ "$ESTABLISHED_COUNT" -gt 0 ]; then
        STATUS="${GREEN}●${NC} HEALTHY - API CONNECTED"
    elif [ "$LISTEN_COUNT" -gt 0 ]; then
        STATUS="${BLUE}●${NC} READY - WAITING FOR CONNECTIONS"
    else
        STATUS="${YELLOW}●${NC} GATEWAY NOT RUNNING"
    fi

    echo -e "${BOLD}Status:${NC} $STATUS"
    echo ""
    echo -e "${CYAN}Press Ctrl+C to stop monitoring...${NC}"

    sleep $INTERVAL
done
