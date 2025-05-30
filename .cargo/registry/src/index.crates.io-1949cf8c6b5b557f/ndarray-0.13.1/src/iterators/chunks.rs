use crate::imp_prelude::*;
use crate::ElementsBase;
use crate::ElementsBaseMut;
use crate::IntoDimension;
use crate::{Layout, NdProducer};

impl_ndproducer! {
    ['a, A, D: Dimension]
    [Clone => 'a, A, D: Clone ]
    ExactChunks {
        base,
        chunk,
        inner_strides,
    }
    ExactChunks<'a, A, D> {
        type Item = ArrayView<'a, A, D>;
        type Dim = D;

        unsafe fn item(&self, ptr) {
            ArrayView::new_(ptr, self.chunk.clone(),
                            self.inner_strides.clone())
        }
    }
}

type BaseProducerRef<'a, A, D> = ArrayView<'a, A, D>;
type BaseProducerMut<'a, A, D> = ArrayViewMut<'a, A, D>;

/// Exact chunks producer and iterable.
///
/// See [`.exact_chunks()`](../struct.ArrayBase.html#method.exact_chunks) for more
/// information.
//#[derive(Debug)]
pub struct ExactChunks<'a, A, D> {
    base: BaseProducerRef<'a, A, D>,
    chunk: D,
    inner_strides: D,
}

impl<'a, A, D: Dimension> ExactChunks<'a, A, D> {
    /// Creates a new exact chunks producer.
    ///
    /// **Panics** if any chunk dimension is zero
    pub(crate) fn new<E>(mut a: ArrayView<'a, A, D>, chunk: E) -> Self
    where
        E: IntoDimension<Dim = D>,
    {
        let chunk = chunk.into_dimension();
        ndassert!(
            a.ndim() == chunk.ndim(),
            concat!(
                "Chunk dimension {} does not match array dimension {} ",
                "(with array of shape {:?})"
            ),
            chunk.ndim(),
            a.ndim(),
            a.shape()
        );
        for i in 0..a.ndim() {
            a.dim[i] /= chunk[i];
        }
        let inner_strides = a.raw_strides();
        a.strides *= &chunk;

        ExactChunks {
            base: a,
            chunk,
            inner_strides,
        }
    }
}

impl<'a, A, D> IntoIterator for ExactChunks<'a, A, D>
where
    D: Dimension,
    A: 'a,
{
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = ExactChunksIter<'a, A, D>;
    fn into_iter(self) -> Self::IntoIter {
        ExactChunksIter {
            iter: self.base.into_elements_base(),
            chunk: self.chunk,
            inner_strides: self.inner_strides,
        }
    }
}

/// Exact chunks iterator.
///
/// See [`.exact_chunks()`](../struct.ArrayBase.html#method.exact_chunks) for more
/// information.
pub struct ExactChunksIter<'a, A, D> {
    iter: ElementsBase<'a, A, D>,
    chunk: D,
    inner_strides: D,
}

impl_ndproducer! {
    ['a, A, D: Dimension]
    [Clone => ]
    ExactChunksMut {
        base,
        chunk,
        inner_strides,
    }
    ExactChunksMut<'a, A, D> {
        type Item = ArrayViewMut<'a, A, D>;
        type Dim = D;

        unsafe fn item(&self, ptr) {
            ArrayViewMut::new_(ptr,
                               self.chunk.clone(),
                               self.inner_strides.clone())
        }
    }
}

/// Exact chunks producer and iterable.
///
/// See [`.exact_chunks_mut()`](../struct.ArrayBase.html#method.exact_chunks_mut)
/// for more information.
//#[derive(Debug)]
pub struct ExactChunksMut<'a, A, D> {
    base: BaseProducerMut<'a, A, D>,
    chunk: D,
    inner_strides: D,
}

impl<'a, A, D: Dimension> ExactChunksMut<'a, A, D> {
    /// Creates a new exact chunks producer.
    ///
    /// **Panics** if any chunk dimension is zero
    pub(crate) fn new<E>(mut a: ArrayViewMut<'a, A, D>, chunk: E) -> Self
    where
        E: IntoDimension<Dim = D>,
    {
        let chunk = chunk.into_dimension();
        ndassert!(
            a.ndim() == chunk.ndim(),
            concat!(
                "Chunk dimension {} does not match array dimension {} ",
                "(with array of shape {:?})"
            ),
            chunk.ndim(),
            a.ndim(),
            a.shape()
        );
        for i in 0..a.ndim() {
            a.dim[i] /= chunk[i];
        }
        let inner_strides = a.raw_strides();
        a.strides *= &chunk;

        ExactChunksMut {
            base: a,
            chunk,
            inner_strides,
        }
    }
}

impl<'a, A, D> IntoIterator for ExactChunksMut<'a, A, D>
where
    D: Dimension,
    A: 'a,
{
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = ExactChunksIterMut<'a, A, D>;
    fn into_iter(self) -> Self::IntoIter {
        ExactChunksIterMut {
            iter: self.base.into_elements_base(),
            chunk: self.chunk,
            inner_strides: self.inner_strides,
        }
    }
}

impl_iterator! {
    ['a, A, D: Dimension]
    [Clone => 'a, A, D: Clone]
    ExactChunksIter {
        iter,
        chunk,
        inner_strides,
    }
    ExactChunksIter<'a, A, D> {
        type Item = ArrayView<'a, A, D>;

        fn item(&mut self, elt) {
            unsafe {
                ArrayView::new_(
                    elt,
                    self.chunk.clone(),
                    self.inner_strides.clone())
            }
        }
    }
}

impl_iterator! {
    ['a, A, D: Dimension]
    [Clone => ]
    ExactChunksIterMut {
        iter,
        chunk,
        inner_strides,
    }
    ExactChunksIterMut<'a, A, D> {
        type Item = ArrayViewMut<'a, A, D>;

        fn item(&mut self, elt) {
            unsafe {
                ArrayViewMut::new_(
                    elt,
                    self.chunk.clone(),
                    self.inner_strides.clone())
            }
        }
    }
}

/// Exact chunks iterator.
///
/// See [`.exact_chunks_mut()`](../struct.ArrayBase.html#method.exact_chunks_mut)
/// for more information.
pub struct ExactChunksIterMut<'a, A, D> {
    iter: ElementsBaseMut<'a, A, D>,
    chunk: D,
    inner_strides: D,
}
