#!/bin/bash
# filepath: pr-2/topic-4/copy_to_remote.sh

set -e

echo "=========================================="
echo "Copying topic-4 to Remote Server"
echo "=========================================="
echo ""

# Configuration
REMOTE_USER="user02"
REMOTE_HOST="152.118.31.24"
REMOTE_PATH="/mnt/sharedfolder/user02/topik-4"
LOCAL_PATH="$(dirname "$0")"

# Check if .gitignore exists
if [ ! -f "$LOCAL_PATH/.gitignore" ]; then
    echo "Warning: .gitignore not found in $LOCAL_PATH"
fi

echo "Source: $LOCAL_PATH"
echo "Destination: $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
echo ""

# Build exclude parameters from .gitignore
EXCLUDE_PARAMS=""
if [ -f "$LOCAL_PATH/.gitignore" ]; then
    while IFS= read -r line; do
        # Skip empty lines and comments
        if [ -n "$line" ] && [ "${line:0:1}" != "#" ]; then
            EXCLUDE_PARAMS="$EXCLUDE_PARAMS --exclude=$line"
        fi
    done < "$LOCAL_PATH/.gitignore"
fi

# Additional excludes
EXCLUDE_PARAMS="$EXCLUDE_PARAMS --exclude=.git --exclude=.gitignore --exclude=copy_to_remote.sh"

echo "Syncing files..."
echo ""

# Use rsync to copy files
rsync -avz --progress \
    $EXCLUDE_PARAMS \
    "$LOCAL_PATH/" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/"

echo ""
echo "=========================================="
echo "Copy completed successfully!"
echo "=========================================="
echo ""
echo "To connect to remote server:"
echo "ssh $REMOTE_USER@$REMOTE_HOST"
echo ""
echo "Files location on remote:"
echo "cd $REMOTE_PATH"