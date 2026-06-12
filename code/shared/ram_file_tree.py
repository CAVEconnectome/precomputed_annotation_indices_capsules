import fnmatch
from typing import Dict, List, Union, Any

class RAMFileTree:
    """
    An in-memory file directory tree where directories are dictionaries
    and files are strings containing data.
    """
    
    def __init__(self, root: Dict[str, Any] = None):
        """
        Initialize the file tree with an optional root dictionary.
        
        Args:
            root: Dictionary representing the root directory structure
        """
        self.root = root if root is not None else {}
    
    def glob(self, pattern: str) -> List[str]:
        """
        Find all paths matching the glob pattern.
        
        Args:
            pattern: Glob pattern (supports * and ? wildcards)
            
        Returns:
            List of matching paths as slash-separated strings
        """
        matches = []
        self._glob_recursive(self.root, pattern.split('/'), [], matches)
        return sorted(matches)
    
    def prefix_glob(self, pattern: str) -> List[str]:
        return self.glob(pattern)
    
    def _glob_recursive(self, current: Dict, pattern_parts: List[str], 
                       path_parts: List[str], matches: List[str]):
        """
        Recursively search for glob matches.
        
        Args:
            current: Current directory dictionary
            pattern_parts: Remaining parts of the pattern to match
            path_parts: Current path being built
            matches: List to accumulate matching paths
        """
        if not pattern_parts:
            return
        
        pattern = pattern_parts[0]
        remaining_pattern = pattern_parts[1:]
        
        for key, value in current.items():
            if fnmatch.fnmatch(key, pattern):
                new_path = path_parts + [key]
                
                if not remaining_pattern:
                    # Pattern is complete, add this match
                    matches.append('/'.join(new_path))
                elif isinstance(value, dict):
                    # Continue matching in subdirectory
                    self._glob_recursive(value, remaining_pattern, new_path, matches)
    
    def get(self, path: str) -> str:
        """
        Retrieve the string data at the specified path.
        
        Args:
            path: Slash-separated path to the file
            
        Returns:
            String data at the path
            
        Raises:
            ValueError: If path doesn't exist or points to a directory
        """
        parts = path.split('/')
        current = self.root
        
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                raise ValueError(f"Path does not exist: {path}")
            current = current[part]
            if not isinstance(current, dict):
                raise ValueError(f"Path component is not a directory: {'/'.join(parts[:i+1])}")
        
        final_key = parts[-1]
        if final_key not in current:
            raise ValueError(f"Path does not exist: {path}")
        
        value = current[final_key]
        if isinstance(value, dict):
            raise ValueError(f"Path points to a directory, not a file: {path}")
        
        return value
    
    def move(self, source: str, destination: str):
        """
        Move a file or directory from source path to destination path.
        
        Args:
            source: Source path
            destination: Destination path
            
        Raises:
            ValueError: If source path doesn't exist
        """
        # Get the item at source
        source_parts = source.split('/')
        source_parent = self.root
        
        for part in source_parts[:-1]:
            if part not in source_parent or not isinstance(source_parent[part], dict):
                raise ValueError(f"Source path does not exist: {source}")
            source_parent = source_parent[part]
        
        source_key = source_parts[-1]
        if source_key not in source_parent:
            raise ValueError(f"Source path does not exist: {source}")
        
        # Store the item to move
        item = source_parent[source_key]
        
        # Create destination path if needed and place the item
        dest_parts = destination.split('/')
        dest_parent = self.root
        
        for part in dest_parts[:-1]:
            if part not in dest_parent:
                dest_parent[part] = {}
            elif not isinstance(dest_parent[part], dict):
                raise ValueError(f"Destination path component is not a directory: {part}")
            dest_parent = dest_parent[part]
        
        dest_key = dest_parts[-1]
        dest_parent[dest_key] = item
        
        # Remove from source
        del source_parent[source_key]
    
    def put(self, path: str, data: str):
        """
        Store data at the specified path, creating directories as needed.
        
        Args:
            path: Slash-separated path where data should be stored
            data: String data to store at the path
        """
        parts = path.split('/')
        current = self.root
        
        # Navigate/create all intermediate directories
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                raise ValueError(f"Path component is a file, not a directory: {part}")
            current = current[part]
        
        # Store the data at the final location
        final_key = parts[-1]
        current[final_key] = data
    
    def getsize(self, path: str) -> int:
        """
        Get the length of the string data at the specified path.
        
        Args:
            path: Slash-separated path to the file
            
        Returns:
            Length of the string data at the path
            
        Raises:
            ValueError: If path doesn't exist or points to a directory
        """
        parts = path.split('/')
        current = self.root
        
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                raise ValueError(f"Path does not exist: {path}")
            current = current[part]
            if not isinstance(current, dict):
                raise ValueError(f"Path component is not a directory: {'/'.join(parts[:i+1])}")
        
        final_key = parts[-1]
        if final_key not in current:
            raise ValueError(f"Path does not exist: {path}")
        
        value = current[final_key]
        if isinstance(value, dict):
            raise ValueError(f"Path points to a directory, not a file: {path}")
        
        return len(value)
    
    def get_total_size(self) -> int:
        """
        Get the total size of all data strings in the tree.
        
        Returns:
            Sum of lengths of all terminal data strings
        """
        return self._calculate_size_recursive(self.root)
    
    def _calculate_size_recursive(self, current: Dict) -> int:
        """
        Recursively calculate the total size of all data strings.
        
        Args:
            current: Current dictionary node
            
        Returns:
            Total size of all data strings in this subtree
        """
        total = 0
        for value in current.values():
            if isinstance(value, dict):
                # Recurse into subdirectory
                total += self._calculate_size_recursive(value)
            else:
                # Add size of data string
                total += len(value)
        return total
    
    def delete(self, path: str):
        """
        Delete a file or directory at the specified path.
        
        Args:
            path: Path to delete
            
        Raises:
            ValueError: If path doesn't exist
        """
        parts = path.split('/')
        current = self.root
        
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                raise ValueError(f"Path does not exist: {path}")
            current = current[part]
        
        final_key = parts[-1]
        if final_key not in current:
            raise ValueError(f"Path does not exist: {path}")
        
        del current[final_key]

    @staticmethod
    def test():
        # Create a sample file tree
        tree = RAMFileTree({
            'home': {
                'user': {
                    'documents': {
                        'report.txt': 'Annual report data',
                        'notes.txt': 'Meeting notes'
                    },
                    'photos': {
                        'vacation.jpg': 'Beach photo data'
                    }
                }
            },
            'etc': {
                'config.txt': 'System configuration'
            }
        })
        
        # Test glob
        print("Files matching 'home/user/*/*.txt':")
        print(tree.glob('home/user/*/*.txt'))
        assert tree.glob('home/user/*/*.txt') == ['home/user/documents/notes.txt', 'home/user/documents/report.txt']
        
        # Test get
        print("\nContent of 'home/user/documents/report.txt':")
        print(tree.get('home/user/documents/report.txt'))
        assert tree.get('home/user/documents/report.txt') == "Annual report data"
        
        # Test move
        tree.move('home/user/documents/notes.txt', 'home/user/notes.txt')
        print("\nAfter moving notes.txt:")
        print(tree.glob('home/user/*.txt'))
        assert tree.glob('home/user/*.txt') == ['home/user/notes.txt']
        
        # Test put
        tree.put('home/user/projects/python/script.py', 'print("Hello World")')
        print("\nAfter adding new file with put():")
        print(tree.glob('home/user/projects/*/*'))
        print(tree.get('home/user/projects/python/script.py'))
        assert tree.glob('home/user/projects/*/*') == ['home/user/projects/python/script.py']
        assert tree.get('home/user/projects/python/script.py') == 'print("Hello World")'
    
        # Test getsize
        print("\nSize of 'etc/config.txt':")
        print(tree.getsize('etc/config.txt'), "characters")
        assert tree.getsize('etc/config.txt') == 20
    
        # Test get_total_size
        print("\nTotal size of all files in tree:")
        print(tree.get_total_size(), "characters")
        assert tree.get_total_size() == 87
        
        # Test delete
        tree.delete('home/user/photos')
        print("\nAfter deleting photos directory:")
        print(tree.glob('home/user/*'))
        assert tree.glob('home/user/*') == ['home/user/documents', 'home/user/notes.txt', 'home/user/projects']

        print("\nAll tests passed")
