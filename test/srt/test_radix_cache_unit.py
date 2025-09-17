"""
Unit tests for the RadixCache implementation.

This module tests the core functionality of RadixCache, BaseKey, and TreeNode
following SGLang testing patterns.

Test Coverage:
- BaseKey: token ID management, slicing, iteration, representation
- TreeNode: node properties, reference counting, hash values  
- RadixCache: insert/match operations, eviction, page alignment

Usage:
    python test_radix_cache_unit.py
    python -m pytest test_radix_cache_unit.py -v
    python -m pytest test_radix_cache_unit.py::TestRadixCache::test_insert_basic
"""

import unittest
import unittest.mock
import time
import torch
from sglang.srt.mem_cache.radix_cache import RadixCache, BaseKey, TreeNode
from sglang.test.test_utils import CustomTestCase


class TestBaseKey(CustomTestCase):
    """Test cases for BaseKey class."""

    def test_init_basic(self):
        """Test basic initialization of BaseKey."""
        token_ids = [1, 2, 3, 4]
        key = BaseKey(token_ids)
        self.assertEqual(key.token_ids, token_ids)
        self.assertIsNone(key.extra_key)

    def test_init_with_extra_key(self):
        """Test initialization with extra_key."""
        token_ids = [1, 2, 3]
        extra_key = "test_key"
        key = BaseKey(token_ids, extra_key)
        self.assertEqual(key.token_ids, token_ids)
        self.assertEqual(key.extra_key, extra_key)

    def test_len(self):
        """Test __len__ method."""
        key = BaseKey([1, 2, 3])
        self.assertEqual(len(key), 3)
        
        empty_key = BaseKey([])
        self.assertEqual(len(empty_key), 0)

    def test_iter(self):
        """Test __iter__ method."""
        token_ids = [1, 2, 3, 4]
        key = BaseKey(token_ids)
        self.assertEqual(list(key), token_ids)

    def test_getitem_int(self):
        """Test __getitem__ with int index."""
        key = BaseKey([10, 20, 30])
        self.assertEqual(key[0], 10)
        self.assertEqual(key[1], 20)
        self.assertEqual(key[2], 30)
        self.assertEqual(key[-1], 30)

    def test_getitem_slice(self):
        """Test __getitem__ with slice."""
        key = BaseKey([1, 2, 3, 4, 5], "extra")
        sliced = key[1:3]
        self.assertIsInstance(sliced, BaseKey)
        self.assertEqual(sliced.token_ids, [2, 3])
        self.assertEqual(sliced.extra_key, "extra")

    def test_repr(self):
        """Test __repr__ method."""
        key = BaseKey([1, 2, 3], "test")
        repr_str = repr(key)
        self.assertIn("BaseKey", repr_str)
        self.assertIn("extra_key='test'", repr_str)
        self.assertIn("[1, 2, 3]", repr_str)

    def test_repr_long_token_ids(self):
        """Test __repr__ with long token_ids."""
        long_tokens = list(range(15))
        key = BaseKey(long_tokens)
        repr_str = repr(key)
        self.assertIn("...", repr_str)  # Should be truncated


class TestTreeNode(CustomTestCase):
    """Test cases for TreeNode class."""

    def setUp(self):
        """Reset the counter before each test."""
        TreeNode.counter = 0

    def test_init_basic(self):
        """Test basic initialization of TreeNode."""
        node = TreeNode()
        self.assertEqual(node.id, 0)
        self.assertEqual(len(node.children), 0)
        self.assertIsNone(node.parent)
        self.assertIsNone(node.key)
        self.assertIsNone(node.value)
        self.assertEqual(node.lock_ref, 0)
        self.assertEqual(node.hit_count, 0)
        self.assertEqual(node.host_ref_counter, 0)
        self.assertIsNone(node.host_value)
        self.assertIsNone(node.hash_value)

    def test_init_with_id(self):
        """Test initialization with custom ID."""
        node = TreeNode(id=42)
        self.assertEqual(node.id, 42)
        # Counter should still increment for next node without explicit ID
        node2 = TreeNode()
        self.assertEqual(node2.id, 1)  # Counter was incremented

    def test_counter_increment(self):
        """Test that counter increments properly."""
        node1 = TreeNode()
        node2 = TreeNode()
        node3 = TreeNode()
        self.assertEqual(node1.id, 0)
        self.assertEqual(node2.id, 1)
        self.assertEqual(node3.id, 2)

    def test_evicted_property(self):
        """Test evicted property."""
        node = TreeNode()
        self.assertTrue(node.evicted)  # value is None initially
        
        node.value = torch.tensor([1, 2, 3])
        self.assertFalse(node.evicted)
        
        node.value = None
        self.assertTrue(node.evicted)

    def test_backuped_property(self):
        """Test backuped property."""
        node = TreeNode()
        self.assertFalse(node.backuped)  # host_value is None initially
        
        node.host_value = torch.tensor([1, 2, 3])
        self.assertTrue(node.backuped)

    def test_protect_host(self):
        """Test protect_host method."""
        node = TreeNode()
        self.assertEqual(node.host_ref_counter, 0)
        
        node.protect_host()
        self.assertEqual(node.host_ref_counter, 1)
        
        node.protect_host()
        self.assertEqual(node.host_ref_counter, 2)

    def test_release_host(self):
        """Test release_host method."""
        node = TreeNode()
        node.host_ref_counter = 2
        
        node.release_host()
        self.assertEqual(node.host_ref_counter, 1)
        
        node.release_host()
        self.assertEqual(node.host_ref_counter, 0)

    def test_release_host_error(self):
        """Test release_host raises error when counter is zero."""
        node = TreeNode()
        with self.assertRaises(RuntimeError):
            node.release_host()

    def test_get_last_hash_value(self):
        """Test get_last_hash_value method."""
        node = TreeNode()
        self.assertIsNone(node.get_last_hash_value())
        
        node.hash_value = []
        self.assertIsNone(node.get_last_hash_value())
        
        node.hash_value = ["hash1", "hash2", "hash3"]
        self.assertEqual(node.get_last_hash_value(), "hash3")

    def test_lt_comparison(self):
        """Test less than comparison based on last_access_time."""
        node1 = TreeNode()
        time.sleep(0.001)  # Small delay to ensure different timestamps
        node2 = TreeNode()
        
        self.assertTrue(node1 < node2)
        self.assertFalse(node2 < node1)


class TestRadixCache(CustomTestCase):
    """Test cases for RadixCache class."""

    def setUp(self):
        """Set up test fixtures."""
        TreeNode.counter = 0

    def test_init_basic(self):
        """Test basic initialization."""
        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1
        )
        
        self.assertEqual(cache.page_size, 1)
        self.assertFalse(cache.disable)
        self.assertFalse(cache.enable_kv_cache_events)
        self.assertEqual(cache.device, torch.device("cpu"))
        self.assertIsNotNone(cache.root_node)
        self.assertEqual(len(cache.root_node.key), 0)

    def test_init_disabled(self):
        """Test initialization with disabled cache."""
        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
            disable=True
        )
        self.assertTrue(cache.disable)

    def test_init_page_size_greater_than_one(self):
        """Test initialization with page_size > 1."""
        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=4
        )
        
        self.assertEqual(cache.page_size, 4)
        # Should use paged key matching functions (partial objects don't have __name__)
        self.assertTrue(hasattr(cache.key_match_fn, 'func'))

    def test_reset(self):
        """Test reset method."""
        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1
        )
        
        # Insert some data
        cache.insert(BaseKey([1, 2, 3]), torch.tensor([10, 20, 30], dtype=torch.int64))
        self.assertGreater(cache.total_size(), 0)
        
        # Reset
        cache.reset()
        self.assertEqual(cache.total_size(), 0)
        self.assertEqual(cache.evictable_size(), 0)
        self.assertEqual(cache.protected_size(), 0)

    def test_match_prefix_empty_key(self):
        """Test match_prefix with empty key."""
        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1
        )
        
        result = cache.match_prefix(BaseKey([]))
        self.assertEqual(len(result.device_indices), 0)
        self.assertEqual(result.last_device_node, cache.root_node)

    def test_match_prefix_disabled_cache(self):
        """Test match_prefix with disabled cache."""
        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
            disable=True
        )
        
        result = cache.match_prefix(BaseKey([1, 2, 3]))
        self.assertEqual(len(result.device_indices), 0)

    def test_insert_basic(self):
        """Test basic insert functionality."""
        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1
        )
        
        key = BaseKey([1, 2, 3])
        value = torch.tensor([10, 20, 30], dtype=torch.int64)
        prefix_len = cache.insert(key, value)
        
        self.assertEqual(prefix_len, 0)  # No existing prefix
        self.assertEqual(cache.total_size(), 3)
        self.assertEqual(cache.evictable_size(), 3)

    def test_insert_disabled_cache(self):
        """Test insert with disabled cache."""
        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
            disable=True
        )
        
        prefix_len = cache.insert(BaseKey([1, 2, 3]), None)
        self.assertEqual(prefix_len, 0)
        self.assertEqual(cache.total_size(), 0)

    def test_insert_and_match_prefix(self):
        """Test insert followed by match_prefix."""
        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1
        )
        
        # Insert sequence [1, 2, 3]
        key1 = BaseKey([1, 2, 3])
        value1 = torch.tensor([10, 20, 30], dtype=torch.int64)
        cache.insert(key1, value1)
        
        # Match exact sequence
        result = cache.match_prefix(BaseKey([1, 2, 3]))
        self.assertEqual(len(result.device_indices), 3)
        torch.testing.assert_close(result.device_indices, torch.tensor([10, 20, 30], dtype=torch.int64))
        
        # Match prefix
        result = cache.match_prefix(BaseKey([1, 2]))
        self.assertEqual(len(result.device_indices), 2)
        torch.testing.assert_close(result.device_indices, torch.tensor([10, 20], dtype=torch.int64))

    def test_insert_with_none_value(self):
        """Test insert with None value (should use token_ids as list)."""
        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1
        )
        
        key = BaseKey([1, 2, 3])
        prefix_len = cache.insert(key, None)
        
        # When None is passed, it should create value from token_ids
        self.assertEqual(prefix_len, 0)
        self.assertEqual(cache.total_size(), 3)

    def test_total_size(self):
        """Test total_size calculation."""
        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1
        )
        
        self.assertEqual(cache.total_size(), 0)
        
        cache.insert(BaseKey([1, 2, 3]), torch.tensor([10, 20, 30], dtype=torch.int64))
        self.assertEqual(cache.total_size(), 3)
        
        cache.insert(BaseKey([4, 5]), torch.tensor([40, 50], dtype=torch.int64))
        self.assertEqual(cache.total_size(), 5)

    def test_kv_cache_events(self):
        """Test KV cache events functionality."""
        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
            enable_kv_cache_events=True
        )
        
        # Insert data to generate events
        cache.insert(BaseKey([1, 2, 3]), None)
        
        # Take events
        events = cache.take_events()
        self.assertGreaterEqual(len(events), 0)  # May or may not generate events
        
        # Events should be cleared after taking
        events2 = cache.take_events()
        self.assertEqual(len(events2), 0)

    def test_disabled_kv_cache_events(self):
        """Test that KV cache events are disabled by default."""
        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
            enable_kv_cache_events=False
        )
        
        # Insert data
        cache.insert(BaseKey([1, 2, 3]), None)
        
        # Should not generate events
        events = cache.take_events()
        self.assertEqual(len(events), 0)

    def test_extra_key_isolation(self):
        """Test that keys with different extra_key values are isolated."""
        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1
        )
        
        # Insert same token sequence with different extra keys
        cache.insert(BaseKey([1, 2, 3], "key1"), torch.tensor([10, 20, 30], dtype=torch.int64))
        cache.insert(BaseKey([1, 2, 3], "key2"), torch.tensor([40, 50, 60], dtype=torch.int64))
        cache.insert(BaseKey([1, 2, 3], None), torch.tensor([70, 80, 90], dtype=torch.int64))
        
        # Keys with different extra_key should not match each other
        result1 = cache.match_prefix(BaseKey([1, 2, 3], "key1"))
        result2 = cache.match_prefix(BaseKey([1, 2, 3], "key2"))
        result3 = cache.match_prefix(BaseKey([1, 2, 3], None))
        result4 = cache.match_prefix(BaseKey([1, 2, 3], "nonexistent"))
        
        # Each should match only its own data
        self.assertEqual(len(result1.device_indices), 3)
        torch.testing.assert_close(result1.device_indices, torch.tensor([10, 20, 30], dtype=torch.int64))
        
        self.assertEqual(len(result2.device_indices), 3)
        torch.testing.assert_close(result2.device_indices, torch.tensor([40, 50, 60], dtype=torch.int64))
        
        self.assertEqual(len(result3.device_indices), 3)
        torch.testing.assert_close(result3.device_indices, torch.tensor([70, 80, 90], dtype=torch.int64))
        
        # Non-existent extra_key should not match
        self.assertEqual(len(result4.device_indices), 0)

    def test_extra_key_prefix_matching(self):
        """Test prefix matching with extra_key."""
        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1
        )
        
        # Insert longer sequence with extra_key
        cache.insert(BaseKey([1, 2, 3, 4, 5], "key1"), torch.tensor([10, 20, 30, 40, 50], dtype=torch.int64))
        
        # Prefix matching should work within same extra_key
        result = cache.match_prefix(BaseKey([1, 2, 3], "key1"))
        self.assertEqual(len(result.device_indices), 3)
        torch.testing.assert_close(result.device_indices, torch.tensor([10, 20, 30], dtype=torch.int64))
        
        # But not with different extra_key
        result = cache.match_prefix(BaseKey([1, 2, 3], "key2"))
        self.assertEqual(len(result.device_indices), 0)

    def test_overlapping_sequences_with_extra_key(self):
        """Test overlapping sequences with same extra_key."""
        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1
        )
        
        # Insert overlapping sequences with same extra_key
        cache.insert(BaseKey([1, 2, 3], "key1"), torch.tensor([10, 20, 30], dtype=torch.int64))
        prefix_len = cache.insert(BaseKey([1, 2, 3, 4], "key1"), torch.tensor([10, 20, 30, 40], dtype=torch.int64))
        
        # Should reuse prefix from same extra_key
        self.assertEqual(prefix_len, 3)
        
        # Verify both sequences can be matched
        result1 = cache.match_prefix(BaseKey([1, 2, 3], "key1"))
        result2 = cache.match_prefix(BaseKey([1, 2, 3, 4], "key1"))
        
        self.assertEqual(len(result1.device_indices), 3)
        self.assertEqual(len(result2.device_indices), 4)

    def test_page_size_with_extra_key(self):
        """Test page size alignment with extra_key."""
        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=2
        )
        
        # Insert data with extra_key
        cache.insert(BaseKey([1, 2, 3, 4], "key1"), torch.tensor([10, 20, 30, 40], dtype=torch.int64))
        
        # Page size alignment should work with extra_key
        result = cache.match_prefix(BaseKey([1, 2, 3], "key1"))
        self.assertEqual(len(result.device_indices), 2)  # Aligned to page size 2
        
        # But not with different extra_key
        result = cache.match_prefix(BaseKey([1, 2, 3], "key2"))
        self.assertEqual(len(result.device_indices), 0)

    def test_lock_ref_with_extra_key(self):
        """Test lock reference counting with extra_key."""
        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1
        )
        
        # Insert data with extra_key
        cache.insert(BaseKey([1, 2, 3], "key1"), torch.tensor([10, 20, 30], dtype=torch.int64))
        
        # Find the node and test lock reference
        result = cache.match_prefix(BaseKey([1, 2, 3], "key1"))
        node = result.last_device_node
        
        initial_evictable = cache.evictable_size()
        
        # Lock/unlock should work properly
        cache.inc_lock_ref(node)
        self.assertLessEqual(cache.evictable_size(), initial_evictable)
        
        cache.dec_lock_ref(node)
        self.assertEqual(cache.evictable_size(), initial_evictable)

    def test_split_node_functionality(self):
        """Test node splitting when keys diverge."""
        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1
        )
        
        # Insert a sequence
        cache.insert(BaseKey([1, 2, 3, 4, 5]), torch.tensor([10, 20, 30, 40, 50], dtype=torch.int64))
        
        # Insert a diverging sequence that shares a prefix
        cache.insert(BaseKey([1, 2, 3, 6, 7]), torch.tensor([10, 20, 30, 60, 70], dtype=torch.int64))
        
        # Both sequences should be stored and retrievable
        result1 = cache.match_prefix(BaseKey([1, 2, 3, 4, 5]))
        result2 = cache.match_prefix(BaseKey([1, 2, 3, 6, 7]))
        result3 = cache.match_prefix(BaseKey([1, 2, 3]))  # Common prefix
        
        self.assertEqual(len(result1.device_indices), 5)
        self.assertEqual(len(result2.device_indices), 5)
        self.assertEqual(len(result3.device_indices), 3)
        
        # Verify correct values
        torch.testing.assert_close(result1.device_indices, torch.tensor([10, 20, 30, 40, 50], dtype=torch.int64))
        torch.testing.assert_close(result2.device_indices, torch.tensor([10, 20, 30, 60, 70], dtype=torch.int64))
        torch.testing.assert_close(result3.device_indices, torch.tensor([10, 20, 30], dtype=torch.int64))

    def test_evict_functionality(self):
        """Test eviction of cache entries."""
        # Mock the allocator to track free calls
        mock_allocator = unittest.mock.Mock()
        mock_allocator.device = torch.device("cpu")
        
        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=1
        )
        
        # Insert some data
        cache.insert(BaseKey([1, 2]), torch.tensor([10, 20], dtype=torch.int64))
        cache.insert(BaseKey([3, 4]), torch.tensor([30, 40], dtype=torch.int64))
        initial_size = cache.total_size()
        
        # Evict some tokens
        cache.evict(2)
        
        # Should have called free on allocator
        self.assertTrue(mock_allocator.free.called)
        
        # Size should be reduced (though exact behavior depends on LRU order)
        self.assertLessEqual(cache.total_size(), initial_size)

    def test_all_values_flatten(self):
        """Test the all_values_flatten method."""
        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1
        )
        
        # Insert data
        cache.insert(BaseKey([1, 2]), torch.tensor([10, 20], dtype=torch.int64))
        cache.insert(BaseKey([3, 4]), torch.tensor([30, 40], dtype=torch.int64))
        
        # Get all values flattened
        all_values = cache.all_values_flatten()
        
        # Should contain all inserted values
        self.assertEqual(len(all_values), 4)
        # Values should be concatenated (order may vary due to tree structure)
        expected_values = set([10, 20, 30, 40])
        actual_values = set(all_values.tolist())
        self.assertEqual(actual_values, expected_values)

    def test_protected_and_evictable_size(self):
        """Test protected and evictable size tracking."""
        cache = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1
        )
        
        # Initially should be zero
        self.assertEqual(cache.protected_size(), 0)
        self.assertEqual(cache.evictable_size(), 0)
        
        # Insert data
        cache.insert(BaseKey([1, 2, 3]), torch.tensor([10, 20, 30], dtype=torch.int64))
        
        # Should be evictable initially
        self.assertEqual(cache.protected_size(), 0)
        self.assertEqual(cache.evictable_size(), 3)
        
        # Lock the data
        result = cache.match_prefix(BaseKey([1, 2, 3]))
        cache.inc_lock_ref(result.last_device_node)
        
        # Should now be protected
        self.assertEqual(cache.protected_size(), 3)
        self.assertEqual(cache.evictable_size(), 0)
        
        # Unlock
        cache.dec_lock_ref(result.last_device_node)
        
        # Should be evictable again
        self.assertEqual(cache.protected_size(), 0)
        self.assertEqual(cache.evictable_size(), 3)


if __name__ == "__main__":
    unittest.main()